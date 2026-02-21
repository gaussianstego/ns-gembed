import sys
import os
import subprocess
import time

from random import SystemRandom
from bisect import bisect_left
import math
import numpy as np

import rawpy as rp
import piexif
from pidng.core import RAW2DNG, DNGTags, Tag

from reedsolo import RSCodec
from kyber_py_mod.ml_kem import ML_KEM_768
from Cryptodome.Cipher import AES
from Cryptodome.Random.random import getrandbits
from Cryptodome.Random import get_random_bytes
from hashlib import sha3_512
import bz2
import struct

import pymp
from numba import njit

# SHA3-512 as PRG. Apply in reproducing G-embedding of Kyber ctxt for CCA check
class PRG:
    def __init__(self, seed): # seed is a byte sequence here
        self.seed = seed
        self.counter = 0
    def random(self):
        digest = sha3_512(self.seed + self.counter.to_bytes(8)).digest()
        int_rep = struct.unpack('<Q', digest[:8])[0]
        self.counter  = self.counter + 1
        return int_rep / (2**64 - 1)
    
# Pei10. Discrete Gaussian sampler. Extracted and ported from OpenFHE
# Added the case for s = 0, and removed floor/ceil for "mean" as our mean is uint
# The "search" is the randomness. Separated from this func. for non-Python JIT speedup
@njit
def discreteGaussian(mean, s, search):
    b_mean = int(mean) # our mean is uint
    if s == 0:
        return b_mean

    b_std = s / math.sqrt(2*math.pi)
    acc = 1e-17
    fin = math.ceil(b_std * math.sqrt(-2 * math.log(acc)))
    mean = float(mean - b_mean)

    variance = b_std*b_std
    cusum = 0
    for x in range(-fin, fin+1):
        cusum = cusum + math.exp(-(x-mean)*(x-mean)/(2*variance))
    b_a = 1/cusum

    m_vals = np.empty(2*fin+2)
    ind = 0
    for i in range(-fin, fin+1):
        m_vals[ind] = b_a * math.exp(-(i-mean)*(i-mean)/(2*variance))
        ind = ind + 1

    m_vals[2*fin+1] = 0
    for i in range(1, 2*fin+2):
        m_vals[i] = m_vals[i] + m_vals[i-1]
    
    ans = np.searchsorted(m_vals, search)

    if ans == len(m_vals):
        raise Exception("Peikert lookup table error. Should not reach here") 
    return ans - fin + b_mean

# Box-Muller transform for gaussian samples
def BoxMuller(rng):
    u1 = rng.random()
    u2 = rng.random()
    r = math.sqrt(-2*math.log(u1))
    theta = 2*math.pi*u2
    return r*math.cos(theta), r*math.sin(theta)

# i.i.d. N(0, 1) k-vector
def multiGaussianStd(k, rng):
    ret = np.empty(k)
    for i in range(k//2):
        ret[2*i], ret[2*i+1] = BoxMuller(rng)
    if k & 1:
        ret[k-1], _ = BoxMuller(rng)
    return ret

# Multivariate (continuous) Gaussian. Apply in gadget sampling
def multiGaussian(sqrtCov, rng): # cov matrix in sense of Gaussian parameters
    ret = multiGaussianStd(sqrtCov.shape[0], rng)
    return (sqrtCov / math.sqrt(2*math.pi)) @ ret

# Least k sig. symbols of u in base b
def int_change_base(u, b, k):
    ret = np.zeros((k,), dtype=np.int64)
    for i in range(k):
        rem = u % b
        ret[i] = rem
        u = u // b
    return ret

# Subroutine for randomization in gadget decomposition
def gInv_pre(beta, ell):
    # y <-$ {-1, 0}^ell, z = Sy
    y = np.array([-int(digit) for digit in bin(getrandbits(ell))[2:].zfill(ell)], dtype=np.int64)
    z = np.empty((ell,), dtype=np.int64)
    z[0] = y[0]*beta
    z[1:] = y[1:]*beta - y[:-1]
    return y, z

# JLP21, zero-mean gadget decomposition for q = beta^ell
def gInv_fit(u, beta, ell, z):
    return int_change_base(u, beta, ell) + z

# JLP23, zero-mean gadget decomposition for q < beta^ell where ell >= 2
def gInv_general(u, beta, ell, q, y, z):
    u = u - (y[ell-1] != 0) * q
    a = u // (beta ** (ell-1))
    u = u % (beta ** (ell-1))
    ret = np.empty((ell,), dtype=np.int64)
    ret[:-1] = gInv_fit(u, beta, ell-1, z[:-1])
    ret[-1] = a + (y[ell-2] != 0)
    return ret

# Zero-mean gadget decomposition
def gInv(u, beta, ell, k):
    # assumme 0 <= u < q = 2^k <= beta^ell
    if ell == 1:
        return np.array([u - beta*getrandbits(1)], dtype=np.int64)
    q = 1 << k # q is 2^num_of_bits_per_g
    y, z = gInv_pre(beta, ell)
    if q == beta**ell:
        return gInv_fit(u, beta, ell, z)
    return gInv_general(u, beta, ell, q, y, z)

# Pei10 Algo 1. Gadget sampling
# Gadget vector [1, beta, ..., beta^(ell-1)], mod 2 = 2^k, where k = floor(ell lg beta)
# Assume precomputed B1 = basis, B1inv = inverse of B1, B1sq = B1 B1^T
# Requirement: sigma > r^2 B1sq, where sigma is diag(Gaussian parameter of pixel^2)
def GadgetGaussian(c, sigma, beta, ell, k, B1, B1inv, B1sq, rng = SystemRandom()):
    cc = gInv(c, beta, ell, k)
    # MP12 suggests r ~= math.sqrt(math.log(2/eps)/math.pi)
    # where eps = stat. err. by each rand.-rounding for ZZ
    # The value of r for eps = 2^-71 is still <= 4
    # This eps is adopted by OpenFHE in Pei10 discrete Gaussian sampler
    r = 4
    sigma2 = sigma - r * r * B1sq
    B2 = np.linalg.cholesky(sigma2)
    x2 = multiGaussian(B2, rng)
    v = B1inv @ (cc - x2)
    for i in range(v.shape[0]):
        search = rng.random()
        v[i] = math.floor(v[i]) + discreteGaussian(-(v[i]-math.floor(v[i])), r, search)
    x = cc - B1 @ v
    return x.astype(np.int64)

# Compute the basis of the kernel gadget lattice
# Gadget vector [1, beta, ..., beta^(ell-1)], mod 2 = 2^k, where k = floor(ell lg beta)
def GadgetBasis(beta, ell, k):
    Sq = np.diag(beta*np.ones(ell, dtype=np.int64))+np.diag(-np.ones(ell-1, dtype=np.int64),-1)
    q = 1 << k
    last_col = int_change_base(q, beta, ell)
    if np.any(last_col):
        Sq[:, ell-1] = last_col
    return Sq

# Upper bound in GPV08. eta <= bl * sqrt(ln(2n(1+1/eps))/pi)
# Kernel gadget lattice GSO norm bl <= sqrt(1+beta*beta) extended from MP12
# Set eps = 2^-71 (match with that used in GadgetGaussian)
def smoothing_upper_bound(beta, ell):
    return math.sqrt(1+beta*beta)*math.sqrt((math.log(1+2**71)+math.log(2*ell))/math.pi) 

# Obtain ISO model (Gaussian parameter for each pixel)
# REMARKS: The output should depend on the camera. Example obtained from WYYL24
def cal_s(f, to_iso):
    # assume from ISO100
    reg100 = np.array([[0.0196348805545999,-3.8478674996469664],
         [0.018671483520605516,-2.504314778156669],
         [0.015967546257640082,-0.3380370307454612]])

    reg200 = np.array([[0.01912681812106003,3.215158817704129],
         [0.01827197101088266,5.792569142588377],
         [0.0157663555390529,6.937844841228667]])

    reg400 = np.array([[0.02013328609041487,12.702013737949514],
         [0.018806856653020766,18.103601960512847],
         [0.0170144047734458,17.430272077032157]])

    reg800 = np.array([[0.02271730241228529,25.187734401059473],
        [0.02051570386411668,35.321711581090085],
        [0.0191944814148252,30.47819459961211]])
    match to_iso:
        case 200:
            reg = reg200
        case 400:
            reg = reg400
        case 800:
            reg = reg800
        case _:
            raise Exception("ISO data not implemented")
    reg = reg-reg100
    raw = f.raw_image
    s = np.empty(np.shape(raw))
    y_pattern_size = f.raw_pattern.shape[0]
    x_pattern_size = f.raw_pattern.shape[1]
    for y in range(y_pattern_size):
        for x in range(x_pattern_size):
            match f.color_desc[f.raw_pattern[y,x]]:
                case 82: # R
                    color = 0
                case 71: # G
                    color = 1
                case 66: #B
                    color = 2
            s[y:None:y_pattern_size, x:None:x_pattern_size] = reg[color,0]*raw[y:None:y_pattern_size, x:None:x_pattern_size]+reg[color,1]
    # obtain Gaussian parameter
    s = s*np.sqrt(2*np.pi)
    return s.astype(np.float32)

# Save as DNG
def save_dng(raw, path, filename):
    height, width = raw.shape
    bit = 16
    pattern = "RGGB"
    CFAPattern = ["RGB".index(c) for c in pattern]
    t = DNGTags()
    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.BitsPerSample, bit)
    t.set(Tag.CFARepeatPatternDim, [2, 2])
    t.set(Tag.CFAPattern, CFAPattern)
    t.set(Tag.BlackLevel, 512)
    t.set(Tag.WhiteLevel, 16380)
    t.set(Tag.DNGVersion, [1, 4, 0, 0])
    t.set(Tag.PhotometricInterpretation, 32803)
    t.set(Tag.PreviewColorSpace, 2)
    t.set(Tag.Orientation, 1)
    raw2dng = RAW2DNG()
    raw2dng.options(
        tags=t,
        path=path,
        compress=False,
    )
    raw2dng.convert(
        raw,
        filename=filename,
    )

# ECC permutation. Takes time to precompute
def load_eccperm(num_of_g_chunks, num_of_bits_per_g):
    filename = 'perm/eccperm_'+str(num_of_g_chunks)+'_'+str(num_of_bits_per_g)+'.npy'
    if not os.path.exists(filename):
        ecc_perm = np.repeat(np.arange(num_of_g_chunks, dtype=np.int32), num_of_bits_per_g)
        prng = np.random.default_rng(1)
        # Use consecutive pixels for a single ECC symbol
        gp_of_bytes = num_of_g_chunks*num_of_bits_per_g // 8
        gp_ecc_perm = ecc_perm[:gp_of_bytes*8].reshape((gp_of_bytes, 8))
        prng.shuffle(gp_ecc_perm)
        with open(filename, 'wb') as f:
            np.save(f, ecc_perm)
    else:
        with open(filename, 'rb') as f:
            ecc_perm = np.load(f)
    return ecc_perm

# Generate ECC permutation files
def gen_eccperm(num_pixel_main):
    list_g_params = []
    for ell in range(1,max_ell+1):
        num_of_g_chunks = num_pixel_main // ell # each G-embedding takes ell pixels
        last_num_of_bits_per_g = 0
        for beta in range(2, max_beta+1):
            num_of_bits_per_g = math.floor(ell * math.log2(beta)) # beta^ell >= q = 2^k
            if num_of_bits_per_g != last_num_of_bits_per_g:
                list_g_params.append((num_of_g_chunks, num_of_bits_per_g))
                last_num_of_bits_per_g = num_of_bits_per_g
    list_g_params = np.array(list_g_params, dtype=np.int32)
    with pymp.Parallel() as p:
        for i in p.range(list_g_params.shape[0]):
            num_of_g_chunks = list_g_params[i,0]
            num_of_bits_per_g = list_g_params[i,1]
            load_eccperm(num_of_g_chunks, num_of_bits_per_g)

# Check if Gaussian parameter > smoothing and fulfulling Pei10 Algo1 requirement
@njit
def find_avail_g(s_g_gp, num_rows, eta_upper, B1sq):
    r = 4 # See GadgetGaussian
    ret = np.zeros((num_rows,), dtype=np.uint8)
    for row in range(num_rows):
        if np.any(s_g_gp[row, :] < eta_upper):
            continue
        try:
            np.linalg.cholesky(np.diag(s_g_gp[row, :]*s_g_gp[row, :]) - r*r*B1sq)
        except:
            continue
        ret[row] = 1
    return ret

# Compute embedding rate
def cal_embed_rate(beta, ell, num_pixel_main, avail_g):
    num_of_g_chunks = num_pixel_main // ell # each G-embedding takes ell pixels
    num_of_bits_per_g = math.floor(ell * math.log2(beta)) # beta^ell >= q = 2^k
    num_ecc_cw = num_of_g_chunks * num_of_bits_per_g // (8 * 255) # every ECC codeword has 255 8-bit symbols
    num_ecc_sym = num_ecc_cw * 255
    ecc_perm = load_eccperm(num_of_g_chunks, num_of_bits_per_g)
    ecc_perm = ecc_perm[:num_ecc_sym*8]
    
    sym_err = np.any(~avail_g[(beta-2)*max_ell+(ell-1), ecc_perm].reshape((num_ecc_sym, 8)), axis=1)
    sym_err_count = np.sum(sym_err.reshape((num_ecc_cw, 255)), axis=1)
    max_err_count = np.max(sym_err_count)
    ecc_red = 2*max_err_count
    if ecc_red >= 255:
        return 0, max_err_count
    return (255 - ecc_red)*num_ecc_cw, max_err_count

# Finding ECC permutation bit locations
@njit
def find_ecc_perm_bit_loc(num_of_g_chunks, num_of_bits_per_g, ecc_perm):
    g_chunks_usage_count = np.zeros((num_of_g_chunks,), dtype=np.uint16)
    ecc_perm_bit_loc = np.empty((num_of_g_chunks*num_of_bits_per_g,), dtype=np.uint16)
    for i in range(num_of_g_chunks*num_of_bits_per_g):
        ecc_perm_bit_loc[i] = g_chunks_usage_count[ecc_perm[i]]
        g_chunks_usage_count[ecc_perm[i]] = g_chunks_usage_count[ecc_perm[i]] + 1
    return ecc_perm_bit_loc

# En/decrypt AES
def endecrypt_aes(num_ecc_cw, mesecc_arr, aes_key, start=0):
    ret = pymp.shared.array(mesecc_arr.shape, dtype=np.uint8)
    with pymp.Parallel() as p:
        for i in p.range(start << 4, num_ecc_cw << 4):
            b_count = i & 15
            ecc_count = i >> 4
            cipher = AES.new(aes_key, AES.MODE_ECB) # single block AES for encrypting the counter
            ctxt = cipher.encrypt(i.to_bytes(16)) # CTR mode. Due to random k, no need IV
            for b in range(16):
                b_ind = b_count << 4 | b
                if b_ind == 255:
                    continue
                ret[ecc_count, b_ind] = mesecc_arr[ecc_count, b_ind] ^ ctxt[b]
    return ret

# Generate data for G-embedding
@njit
def gen_gmsg(num_of_g_chunks, num_of_bits_per_g, num_ecc_cw, mesecc_arr, ecc_perm, ecc_perm_bit_loc, rndbits):
    gmsg_arr = np.zeros((num_of_g_chunks,), dtype=np.uint64)
    for i in range(num_of_g_chunks*num_of_bits_per_g):
        if i < num_ecc_cw*255*8:
            ecc_ind = i // (255*8)
            sym_ind = i % (255*8) // 8
            bit_ind = i % (255*8) % 8
            ecc_bit = mesecc_arr[ecc_ind, sym_ind] >> bit_ind & 1
        else:
            ecc_bit = rndbits[i - num_ecc_cw*255*8]
        gmsg_arr[ecc_perm[i]] = gmsg_arr[ecc_perm[i]] | int(ecc_bit) << int(ecc_perm_bit_loc[i])
    return gmsg_arr

# Embedding body
def stego_embedding(raw, s):
    raw_flatten = raw.flatten()
    s_flat = s.flatten()
    possible_master_count = [s.shape[0] // master_height, s.shape[1] // master_width]
    prng = np.random.default_rng(0)
    master_loc = prng.choice(possible_master_count[0] * possible_master_count[1], master_dup, replace=False)
    # Check for available master chunks
    main_map = np.ones(s.shape, dtype=np.uint8)
    master_s = np.empty((master_dup,))
    for i in range(master_dup):
        y = master_loc[i] // possible_master_count[1] * master_height
        x = master_loc[i] % possible_master_count[1] * master_width
        main_map[y:y+master_height, x:x+master_width] = 0
        master_s[i] = np.min(s[y:y+master_height, x:x+master_width])
    master_eta = smoothing_upper_bound(2, 1)
    failed_master = master_s < master_eta
    if np.all(failed_master):
        raise Exception("No available embedding for master chunk")
    total_pixels = s.shape[0]*s.shape[1]
    pixel_id = np.arange(total_pixels).reshape(s.shape)
    num_pixel_main = total_pixels - num_pixel_master
    # Pixel indices without master chunks. Each G-embedding takes ell pixels sequentially
    # It can be permutated if needed
    g_perm = np.flatnonzero(main_map).astype(np.int32)
    # Precompute ECC permutation 
    print("Generating/loading ECC permutation...")
    begin = time.monotonic()
    gen_eccperm(num_pixel_main)
    end = time.monotonic()
    elapsed = end-begin
    print(f"ECC permutation done in {elapsed} seconds")
    # Prepare for searching optimal gadget parameters
    print("Searching for optimal parameters...")
    begin = time.monotonic()
    avail_g = pymp.shared.array(((max_beta-1)*max_ell, num_pixel_main), dtype=bool)
    with pymp.Parallel() as p:
        for i in p.range((max_beta-1)*max_ell):
            beta = i // max_ell + 2
            ell = i % max_ell + 1
            B1 = GadgetBasis(beta, ell, math.floor(ell * math.log2(beta)))
            B1sq = B1 @ np.transpose(B1)
            s_g_gp = s_flat[g_perm[:num_pixel_main // ell * ell]].reshape((num_pixel_main // ell, ell))
            avail_g[i, :num_pixel_main // ell] = find_avail_g(s_g_gp, num_pixel_main // ell, smoothing_upper_bound(beta, ell), B1sq)
    rate_table = pymp.shared.array((max_beta-1, max_ell), dtype=np.int64)
    max_err_table = pymp.shared.array((max_beta-1, max_ell), dtype=np.int64)
    with pymp.Parallel() as p:
        for i in p.range((max_beta-1)*max_ell):
            beta = i // max_ell + 2
            ell = i % max_ell + 1
            rate_table[beta-2, ell-1], max_err_table[beta-2, ell-1] = cal_embed_rate(beta, ell, num_pixel_main, avail_g)
    # Search for optimal gadget parameters
    rate = np.max(rate_table, axis=None)
    (beta, ell) = np.unravel_index(np.argmax(rate_table, axis=None), rate_table.shape)
    err_count = max_err_table[beta, ell]
    beta = beta + 2
    ell = ell + 1
    print(f"Optimal parameters: beta = {beta}, ell = {ell}, rate = {rate}, err = {err_count}")
    end = time.monotonic()
    elapsed = end-begin
    print(f"Optimal parameters found in {elapsed} seconds")
    if rate == 0:
        raise Exception("Too many ECC errors")
    # Compress pixels and Gaussian parameters at Kyber ctxt before embedding. For CCA check
    # This info is part of the message payload
    img_meta = np.empty((master_dup, metadata_offset), dtype=np.uint16)
    s_meta = np.empty((master_dup, metadata_offset), dtype=np.float32)
    s_counter = 0
    for i in range(master_dup):
        if failed_master[i]:
            continue
        y = master_loc[i] // possible_master_count[1] * master_height
        x = master_loc[i] % possible_master_count[1] * master_width
        pixel_loc = pixel_id[y:y+master_height, x:x+master_width].flatten()
        pixel_loc = pixel_loc[:metadata_offset]
        img_meta[s_counter, :] = raw_flatten[pixel_loc]
        s_meta[s_counter, :] = s_flat[pixel_loc]
        s_counter = s_counter + 1
    img_meta = img_meta[:s_counter, :].tobytes()
    s_meta = s_meta[:s_counter, :].tobytes()
    combined_meta = img_meta + s_meta
    img_meta_compressed = bz2.compress(img_meta)
    img_meta_len = len(img_meta_compressed)
    if img_meta_len >= len(img_meta):
        img_meta_compressed = img_meta
        img_meta_len = 0
    s_meta_compressed = bz2.compress(s_meta)
    s_meta_len = len(s_meta_compressed)
    if s_meta_len >= len(s_meta):
        s_meta_compressed = s_meta
        s_meta_len = 0
    use_combined_meta = False
    combined_meta_compressed = bz2.compress(combined_meta)
    combined_meta_len = len(combined_meta_compressed)
    if combined_meta_len >= len(combined_meta):
        combined_meta_compressed = combined_meta
        combined_meta_len = 0
    cca_metadata_len = len(img_meta_compressed) + len(s_meta_compressed)
    if len(combined_meta_compressed) < len(img_meta_compressed) + len(s_meta_compressed):
        cca_metadata_len = len(combined_meta_compressed)
        use_combined_meta = True
    print(f"CCA metadata pixel compression ratio: {len(img_meta)}/{len(img_meta_compressed)} = {len(img_meta)/len(img_meta_compressed)}")
    print(f"CCA metadata s compression ratio: {len(s_meta)}/{len(s_meta_compressed)} = {len(s_meta)/len(s_meta_compressed)}")
    print(f"CCA metadata combined compression ratio: {len(combined_meta)}/{len(combined_meta_compressed)} = {len(combined_meta)/len(combined_meta_compressed)}")
    print(f"Use combined compression = {use_combined_meta}")
    print(f"CCA metadata size = {cca_metadata_len}")
    # Prepare payload
    num_msg_bytes = rate-len(combined_meta_compressed)
    print(f"Supported message size = {num_msg_bytes}")
    if num_msg_bytes < filesize:
        raise Exception("Not enough space to embed")
    if use_combined_meta:
        msg = combined_meta_compressed + ori_msg + b'\0'*(rate-filesize-cca_metadata_len)
    else:
        msg = img_meta_compressed + s_meta_compressed + ori_msg + b'\0'*(rate-filesize-cca_metadata_len)
    # Prepare for embedding data
    ecc_red = 2*err_count
    ecc_msg_len = 255 - ecc_red
    num_ecc_cw = rate // ecc_msg_len
    num_of_g_chunks = num_pixel_main // ell # each G-embedding takes ell pixels
    num_of_bits_per_g = math.floor(ell * math.log2(beta)) # beta^ell >= q = 2^k
    # ECC permutation bit locations
    print(f"Computing ECC bit locations")
    begin = time.monotonic()
    ecc_perm = load_eccperm(num_of_g_chunks, num_of_bits_per_g)
    ecc_perm_bit_loc = find_ecc_perm_bit_loc(num_of_g_chunks, num_of_bits_per_g, ecc_perm)
    end = time.monotonic()
    elapsed = end-begin
    print(f"ECC bit locations computed in {elapsed} seconds")
    # Generate AES key
    aes_key, kyber_rnd = ML_KEM_768.encap_key(ek)
    # ECC encoding
    print(f"ECC encoding...")
    begin = time.monotonic()
    mesecc_arr = pymp.shared.array((num_ecc_cw,255), dtype=np.uint8)
    with pymp.Parallel() as p:
        for ecc_count in p.range(num_ecc_cw):
            if ecc_red == 0:
                mesecc_arr[ecc_count, :] = np.frombuffer(msg[ecc_count*ecc_msg_len:(ecc_count+1)*ecc_msg_len], dtype=np.uint8)
            else:
                rsc = RSCodec(ecc_red)
                mes = msg[ecc_count*ecc_msg_len:(ecc_count+1)*ecc_msg_len]
                mesecc = rsc.encode(mes)
                mesecc_arr[ecc_count, :] = np.array(mesecc, dtype=np.uint8)
    end = time.monotonic()
    elapsed = end-begin
    print(f"ECC encoding done in {elapsed} seconds")
    # AES encrypting the ECC codewords
    print("AES encrypting...")
    begin = time.monotonic()
    mesecc_arr = endecrypt_aes(num_ecc_cw, mesecc_arr, aes_key)
    end = time.monotonic()
    elapsed = end-begin
    print(f"AES encryption done in {elapsed} seconds")
    # Generate data for G-embedding
    print("Preparing data for G-embedding...")
    begin = time.monotonic()
    rndbits_len = (num_of_g_chunks*num_of_bits_per_g - num_ecc_cw*255*8) // 8
    if (num_of_g_chunks*num_of_bits_per_g - num_ecc_cw*255*8) % 8 != 0:
        rndbits_len = rndbits_len + 1
    rndbits = np.unpackbits(np.frombuffer(get_random_bytes(rndbits_len), dtype=np.uint8))
    gmsg_arr = gen_gmsg(num_of_g_chunks, num_of_bits_per_g, num_ecc_cw, mesecc_arr, ecc_perm, ecc_perm_bit_loc, rndbits)
    end = time.monotonic()
    elapsed = end-begin
    print(f"G-embedding data prepared in {elapsed} seconds")
    # G-embedding
    print("G-embedding...")
    begin = time.monotonic()
    B1 = GadgetBasis(beta, ell, num_of_bits_per_g)
    B1inv = np.linalg.inv(B1)
    B1sq = B1 @ np.transpose(B1)
    g_vec = np.power(beta, range(ell))
    stego_img = pymp.shared.array(raw_flatten.shape, dtype=np.uint16)
    oob = pymp.shared.array((1,), dtype=np.uint8)
    with pymp.Parallel() as p:
        for i in p.range(num_of_g_chunks):
            failed_to_embed = True
            pixel_loc = g_perm[i*ell:(i+1)*ell]
            if avail_g[(beta-2)*max_ell+(ell-1), i]:
                sigma = np.diag(s_flat[pixel_loc] * s_flat[pixel_loc])
                temp_stego = raw_flatten[pixel_loc] + GadgetGaussian((gmsg_arr[i] - g_vec @ raw_flatten[pixel_loc]) % (1 << num_of_bits_per_g), sigma, beta, ell, num_of_bits_per_g, B1, B1inv, B1sq)
                if np.any(temp_stego < 0) or np.any(temp_stego > 17204):
                    oob[0] = True
                else:
                    failed_to_embed = False
            if failed_to_embed:
                temp_stego = np.zeros((ell,))
                rng = SystemRandom()
                for ind in range(ell):
                    search = rng.random()
                    temp_stego[ind] = discreteGaussian(raw_flatten[pixel_loc[ind]], s_flat[pixel_loc[ind]], search)
                # oob truncation
                temp_stego[np.where(temp_stego < 0)] = 0
                temp_stego[np.where(temp_stego > 17204)] = 17204
            stego_img[pixel_loc] = temp_stego
        for i in p.range(num_of_g_chunks*ell, num_pixel_main):
            pixel_loc = g_perm[i]
            search = rng.random()
            temp_stego = discreteGaussian(raw_flatten[pixel_loc], s_flat[pixel_loc], search)
            if temp_stego < 0:
                temp_stego = 0
            if temp_stego > 17204:
                temp_stego = 17204
            stego_img[pixel_loc] = temp_stego
    if oob[0]:
        print("Ignored out-of-bound chunk. Potential extraction failure")
    end = time.monotonic()
    elapsed = end-begin
    print(f"G-embedding done in {elapsed} seconds")
    # Generate metadata
    if use_combined_meta:
        metadata = filesize.to_bytes(4) + b'\1' + combined_meta_len.to_bytes(3) + b'\0'*3 + int(beta).to_bytes(1) + int(ell).to_bytes(1) + int(err_count).to_bytes(1) + int(failed_master @ np.pow(2, range(len(failed_master)))).to_bytes(2)
    else:
        metadata = filesize.to_bytes(4) + b'\0' + img_meta_len.to_bytes(3) + s_meta_len.to_bytes(3) + int(beta).to_bytes(1) + int(ell).to_bytes(1) + int(err_count).to_bytes(1) + int(failed_master @ np.pow(2, range(len(failed_master)))).to_bytes(2)
    # Embed metadata and generate unavailable master chunks, i.e., only Kyber ctxt parts are still 0
    print("Embedding metadata...")
    begin = time.monotonic()
    B1 = GadgetBasis(2, 8, 8)
    B1inv = np.linalg.inv(B1)
    B1sq = B1 @ np.transpose(B1)
    g_vec = np.power(2, range(8))
    for i in range(master_dup):
        y = master_loc[i] // possible_master_count[1] * master_height
        x = master_loc[i] % possible_master_count[1] * master_width
        pixel_loc = pixel_id[y:y+master_height, x:x+master_width].flatten()
        temp_stego = pymp.shared.array((master_size,), dtype=np.uint16)
        if not failed_master[i]:
            # fill in metadata
            cipher = AES.new(aes_key, AES.MODE_ECB)
            ctxt = cipher.encrypt((2**128-1-i).to_bytes(16))
            enc_metadata = np.empty((16,), dtype=np.uint8)
            for j in range(16):
                enc_metadata[j] = metadata[j] ^ ctxt[j]
            for j in range(16): # metadata has 16 bytes
                sigma = np.diag(s_flat[pixel_loc[j*8+metadata_offset:(j+1)*8+metadata_offset]] * s_flat[pixel_loc[j*8+metadata_offset:(j+1)*8+metadata_offset]])
                temp_stego[j*8+metadata_offset:(j+1)*8+metadata_offset] = raw_flatten[pixel_loc[j*8+metadata_offset:(j+1)*8+metadata_offset]] + GadgetGaussian((enc_metadata[j] - g_vec @ raw_flatten[pixel_loc[j*8+metadata_offset:(j+1)*8+metadata_offset]]) % 256, sigma, 2, 8, 8, B1, B1inv, B1sq)
            if np.any(temp_stego[metadata_offset:] < 0) or np.any(temp_stego[metadata_offset:] > 17204):
                raise Exception("oob in metadata embedding") # possible solution is to restart the master chunk embedding process
        else:
            with pymp.Parallel() as p:
                for ind in p.range(master_size):
                    rng = SystemRandom()
                    search = rng.random()
                    temp_stego[ind] = discreteGaussian(raw_flatten[pixel_loc[ind]], s_flat[pixel_loc[ind]], search)
        stego_img[pixel_loc] = temp_stego
    end = time.monotonic()
    elapsed = end-begin
    print(f"Metadata embedded in {elapsed} seconds")
    # Generate and embed Kyber ctxt. The stego up to this point is hashed for CCA check
    print("Embedding Kyber...")
    begin = time.monotonic()
    fo_hash = sha3_512(stego_img).digest()
    clist = ML_KEM_768.encaps_modded(ek, kyber_rnd, fo_hash, master_dup)
    prg_seed = ML_KEM_768.encaps_hash(ek, kyber_rnd, fo_hash)
    for i in range(master_dup):
        if failed_master[i]:
            continue
        y = master_loc[i] // possible_master_count[1] * master_height
        x = master_loc[i] % possible_master_count[1] * master_width
        pixel_loc = pixel_id[y:y+master_height, x:x+master_width].flatten()
        pixel_loc = pixel_loc[:metadata_offset]
        temp_stego = np.empty((metadata_offset,), dtype=np.uint16)
        for j in range(1088): # kyber ctxt has 1088 bytes
            prg = PRG(prg_seed+i.to_bytes(8)+j.to_bytes(8))
            sigma = np.diag(s_flat[pixel_loc[j*8:(j+1)*8]] * s_flat[pixel_loc[j*8:(j+1)*8]])
            temp_stego[j*8:(j+1)*8] = raw_flatten[pixel_loc[j*8:(j+1)*8]] + GadgetGaussian((clist[i][j] - g_vec @ raw_flatten[pixel_loc[j*8:(j+1)*8]]) % 256, sigma, 2, 8, 8, B1, B1inv, B1sq, prg)
        if np.any(temp_stego < 0) or np.any(temp_stego > 17204):
            raise Exception("oob in kyber embedding") # possible solution is to restart the master chunk embedding process
        stego_img[pixel_loc] = temp_stego
    end = time.monotonic()
    elapsed = end-begin
    print(f"Kyber embedded in {elapsed} seconds")
    # Done
    stego_img = stego_img.reshape(s.shape)
    return stego_img

# Decrypt metadata
def decrypt_metadata(enc_metadata, aes_key, id):
    metadata = np.empty((16,), dtype=np.uint8)
    cipher = AES.new(aes_key, AES.MODE_ECB)
    ctxt = cipher.encrypt((2**128-1-id).to_bytes(16))
    for j in range(16):
        metadata[j] = enc_metadata[j] ^ ctxt[j]
    return bytes(metadata)

# Extract failure vector from a master chunk and perform sanity check
def metadata_check(metadata_pixel, aes_key, id):
    # extract the failure vector (it can be errorous if AES key is wrong)
    g_vec = np.power(2, range(8))
    enc_metadata = np.empty((16,), dtype=np.uint8)
    for j in range(16):
        enc_metadata[j] = (g_vec @ metadata_pixel[j*8:(j+1)*8])%256
    metadata = decrypt_metadata(enc_metadata, aes_key, id)
    beta = int.from_bytes(metadata[11:12])
    ell = int.from_bytes(metadata[12:13])
    err_count = int.from_bytes(metadata[13:14]) # quick convert, as max err is 127
    failed_master = int.from_bytes(metadata[14:16]) # quick convert, assume sign bit is 0
    # oob value, or the target chunk is marked as failure
    if failed_master >= 1 << master_dup or failed_master < 0 or failed_master >> id & 1 != 0 or \
    beta < 2 or beta > max_beta or ell < 0 or ell > max_ell or err_count < 0 or err_count > 127:
        return False, b''
    return True, metadata

# extract AES ctxt
@njit
def extract_aes(num_ecc_cw, gmsg_arr, ecc_perm, ecc_perm_bit_loc, start=0):
    mesecc_arr = np.zeros((num_ecc_cw,255), dtype=np.uint8)
    for i in range(start*255*8, num_ecc_cw*255*8):
        ecc_ind = i // (255*8)
        sym_ind = i % (255*8) // 8
        bit_ind = i % (255*8) % 8
        ecc_bit = gmsg_arr[ecc_perm[i]] >> ecc_perm_bit_loc[i] & 1
        mesecc_arr[ecc_ind, sym_ind] = mesecc_arr[ecc_ind, sym_ind] | ecc_bit << bit_ind
    return mesecc_arr

# Decode ECC
def decode_ecc(num_ecc_cw, ecc_red, mesecc_arr, start=0):
    ecc_msg_len = 255 - ecc_red
    decmsg_arr = pymp.shared.array((num_ecc_cw, ecc_msg_len), dtype=np.uint8)
    ecc_fail = pymp.shared.array((1,), dtype=np.uint8)
    with pymp.Parallel() as p:
        for ecc_count in p.range(start, num_ecc_cw):
            if ecc_red == 0:
                decmsg_arr[ecc_count, :] = mesecc_arr[ecc_count, :]
            else:
                rsc = RSCodec(ecc_red)
                try:
                    rmes, _, _ = rsc.decode(mesecc_arr[ecc_count, :].tobytes())
                    decmsg_arr[ecc_count, :] = np.array(rmes, dtype=np.uint8)
                except:
                    ecc_fail[0] = 1
    return ecc_fail[0] == 0, decmsg_arr

# CCA check
def cca_check(failed_master, img_meta, s_meta, master_loc, possible_master_count, pixel_id, stego_img, aes_key):
    # Removing Kyber ctxt from stego for FO hash
    main_map = stego_img.copy()
    for i in range(master_dup):
        if failed_master >> i & 1 == 0:
            y = master_loc[i] // possible_master_count[1] * master_height
            x = master_loc[i] % possible_master_count[1] * master_width
            pixel_loc = pixel_id[y:y+master_height, x:x+master_width].flatten()
            main_map[pixel_loc[:metadata_offset]] = 0
    fo_hash = sha3_512(main_map).digest()
    # Check all Kyber ctxt
    s_counter = 0
    B1 = GadgetBasis(2, 8, 8)
    B1inv = np.linalg.inv(B1)
    B1sq = B1 @ np.transpose(B1)
    g_vec = np.power(2, range(8))
    for i in range(master_dup):
        if failed_master >> i & 1 == 0:
            y = master_loc[i] // possible_master_count[1] * master_height
            x = master_loc[i] % possible_master_count[1] * master_width
            pixel_loc = pixel_id[y:y+master_height, x:x+master_width].flatten()
            temp_stego = stego_img[pixel_loc[:metadata_offset]]
            # Verify Kyber
            kyber_ctxt = b''
            for j in range(1088):
                kyber_ctxt = kyber_ctxt + int((g_vec @ temp_stego[j*8:(j+1)*8])%256).to_bytes(1)
            if not ML_KEM_768.decaps_verify(dk, kyber_ctxt, fo_hash, i):
                return False
            # G-embeeding check
            prg_seed = ML_KEM_768.decaps_hash(dk, kyber_ctxt, fo_hash)
            temp_stego_prime = np.empty((metadata_offset,), dtype=np.uint16)
            for j in range(1088): # kyber ctxt has 1088 bytes
                prg = PRG(prg_seed+i.to_bytes(8)+j.to_bytes(8))
                sigma = np.diag(s_meta[s_counter, j*8:(j+1)*8] * s_meta[s_counter, j*8:(j+1)*8])
                try:
                    temp_stego_prime[j*8:(j+1)*8] = img_meta[s_counter, j*8:(j+1)*8] + GadgetGaussian((kyber_ctxt[j] - g_vec @ img_meta[s_counter, j*8:(j+1)*8]) % 256, sigma, 2, 8, 8, B1, B1inv, B1sq, prg)
                except:
                    return False
            if np.any(temp_stego_prime != temp_stego):
                return False
            s_counter = s_counter + 1
    return True

# Extraction body
def stego_extract(raw):
    stego_img = raw.astype(np.uint16).flatten()
    total_pixels = len(stego_img)
    pixel_id = np.arange(total_pixels).reshape(raw.shape)
    possible_master_count = [raw.shape[0] // master_height, raw.shape[1] // master_width]
    prng = np.random.default_rng(0)
    master_loc = prng.choice(possible_master_count[0] * possible_master_count[1], master_dup, replace=False)
    main_map = raw.copy()
    for i in range(master_dup):
        y = master_loc[i] // possible_master_count[1] * master_height
        x = master_loc[i] % possible_master_count[1] * master_width
        pixel_loc = pixel_id[y:y+master_height, x:x+master_width].flatten()
        main_map[y:y+master_height, x:x+master_width] = 0
    # Pixel indices without master chunks. Each G-embedding takes ell pixels sequentially
    # It can be permutated if needed
    g_perm = np.flatnonzero(main_map).astype(np.int32)
    num_pixel_main = total_pixels - num_pixel_master
    # Extract AES key and metadata, and perform CCA check
    begin = time.monotonic()
    master_dec_success = False
    for i in range(master_dup):
        print(f"Extracting from master chunk {i}...")
        g_vec = np.power(2, range(8))
        y = master_loc[i] // possible_master_count[1] * master_height
        x = master_loc[i] % possible_master_count[1] * master_width
        pixel_loc = pixel_id[y:y+master_height, x:x+master_width].flatten()
        temp_stego = stego_img[pixel_loc]
        # Extract Kyber ctxt
        kyber_ctxt = b''
        for j in range(1088):
            kyber_ctxt = kyber_ctxt + int((g_vec @ temp_stego[j*8:(j+1)*8])%256).to_bytes(1)
        # Decapsulate AES key and extract metadata
        aes_key = ML_KEM_768.decaps_key(dk, kyber_ctxt)
        flag, metadata = metadata_check(temp_stego[metadata_offset:], aes_key, i)
        if flag == False:
            continue
        # Metadata passed sanity check, but it might be incorrect due to wrong AES key
        # Compute CCA metadata size
        beta = int.from_bytes(metadata[11:12])
        ell = int.from_bytes(metadata[12:13])
        err_count = int.from_bytes(metadata[13:14])
        failed_master = int.from_bytes(metadata[14:16])
        use_combined_meta = int.from_bytes(metadata[4:5]) == 1
        num_of_g_chunks = num_pixel_main // ell
        num_of_bits_per_g = math.floor(ell * math.log2(beta))
        ecc_perm = load_eccperm(num_of_g_chunks, num_of_bits_per_g)
        ecc_perm_bit_loc = find_ecc_perm_bit_loc(num_of_g_chunks, num_of_bits_per_g, ecc_perm)
        ecc_msg_len = 255 - 2*err_count
        if use_combined_meta:
            combined_meta_len = int.from_bytes(metadata[5:8])
            combined_meta_has_compressed = True
            if combined_meta_len == 0:
                combined_meta_has_compressed = False
                combined_meta_len = ((master_dup - failed_master.bit_count()) * metadata_offset) * 6 # size of float32 + uint16 = 6
                if combined_meta_len <= 0:
                    continue
        else:
            img_meta_len = int.from_bytes(metadata[5:8])
            img_meta_has_compressed = True
            if img_meta_len == 0:
                img_meta_has_compressed = False
                img_meta_len = ((master_dup - failed_master.bit_count()) * metadata_offset) * 2
                if img_meta_len <= 0:
                    continue
            s_meta_len = int.from_bytes(metadata[8:11])
            s_meta_has_compressed = True
            if s_meta_len == 0:
                s_meta_has_compressed = False
                s_meta_len = (master_dup - failed_master.bit_count()) * metadata_offset * 4
                if s_meta_len <= 0:
                    continue
            combined_meta_len = img_meta_len + s_meta_len
        num_meta_cw = combined_meta_len // ecc_msg_len
        if combined_meta_len % ecc_msg_len != 0:
            num_meta_cw = num_meta_cw + 1
        # Extract CCA metadata
        print("Extracting CCA metadata...")
        ind = np.unique(ecc_perm[:num_meta_cw*255*8])
        g_vec = np.power(beta, range(ell))
        gmsg_arr = pymp.shared.array((num_of_g_chunks,), dtype=np.uint64)
        with pymp.Parallel() as p:
            for j in p.range(len(ind)):
                id = ind[j]
                pixel_loc = g_perm[id*ell:(id+1)*ell]
                gmsg_arr[id] = (g_vec @ stego_img[pixel_loc]) % (1 << num_of_bits_per_g)
        mesecc_arr = extract_aes(num_meta_cw, gmsg_arr, ecc_perm, ecc_perm_bit_loc)
        mesecc_arr = endecrypt_aes(num_meta_cw, mesecc_arr, aes_key)
        ecc_ok, decmsg_arr = decode_ecc(num_meta_cw, 2*err_count, mesecc_arr)
        if not ecc_ok:
            continue
        decmsg_arr = decmsg_arr.tobytes()
        if len(decmsg_arr) < combined_meta_len:
            continue
        img_meta_correct_size = (master_dup - failed_master.bit_count()) * metadata_offset * 2
        s_meta_correct_size = (master_dup - failed_master.bit_count()) * metadata_offset * 4
        if use_combined_meta:
            combined_meta_compressed = decmsg_arr[:combined_meta_len]
            if combined_meta_has_compressed:
                try:
                    combined_meta = bz2.decompress(combined_meta_compressed)
                except:
                    continue
            else:
                combined_meta = combined_meta_compressed
            if len(combined_meta) != img_meta_correct_size + s_meta_correct_size:
                continue
            img_meta = combined_meta[:img_meta_correct_size]
            s_meta = combined_meta[img_meta_correct_size:]
        else:
            img_meta_compressed = decmsg_arr[:img_meta_len]
            s_meta_compressed = decmsg_arr[img_meta_len:combined_meta_len]
            if img_meta_has_compressed:
                try:
                    img_meta = bz2.decompress(img_meta_compressed)
                except:
                    continue
            else:
                img_meta = img_meta_has_compressed
            if len(img_meta) != img_meta_correct_size:
                continue
            if s_meta_has_compressed:
                try:
                    s_meta = bz2.decompress(s_meta_compressed)
                except:
                    continue
            else:
                s_meta = s_meta_compressed
            if len(s_meta) != s_meta_correct_size:
                continue
        img_meta = np.frombuffer(img_meta, dtype=np.uint16)
        img_meta = img_meta.reshape((len(img_meta)//metadata_offset, metadata_offset))
        s_meta = np.frombuffer(s_meta, dtype=np.float32)
        s_meta = s_meta.reshape((len(s_meta)//metadata_offset, metadata_offset))
        # CCA check
        print("Performing CCA check...")
        if cca_check(failed_master, img_meta, s_meta, master_loc, possible_master_count, pixel_id, stego_img, aes_key):
            print("CCA check passed")
            master_dec_success = True
            filesize = int.from_bytes(metadata[:4])
            break
        else:
            print("CCA check failed. Need to try another master chunk")
    end = time.monotonic()
    elapsed = end-begin
    print(f"CCA check finished in {elapsed} seconds")
    if not master_dec_success:
        raise Exception("CCA check failed")
    # Prepare for extraction
    main_data_len = combined_meta_len + filesize
    num_ecc_cw = main_data_len // ecc_msg_len
    if main_data_len % ecc_msg_len != 0:
        num_ecc_cw = num_ecc_cw + 1
    print(f"CCA metadata size = {combined_meta_len}, msg size = {filesize}")
    # G-extraction
    print("G-extracting...")
    print(f"Parameters: beta = {beta}, ell = {ell}, err = {err_count}")
    begin = time.monotonic()
    g_vec = np.power(beta, range(ell))
    gmsg_arr = pymp.shared.array((num_of_g_chunks,), dtype=np.uint32)
    with pymp.Parallel() as p:
        for i in p.range(num_of_g_chunks):
            pixel_loc = g_perm[i*ell:(i+1)*ell]
            gmsg_arr[i] = (g_vec @ stego_img[pixel_loc]) % (1 << num_of_bits_per_g)
    end = time.monotonic()
    elapsed = end-begin
    print(f"G-extraction done in {elapsed} seconds")
    # Extract AES ctxt
    print("Extracting AES ciphertexts...")
    begin = time.monotonic()
    mesecc_arr = extract_aes(num_ecc_cw, gmsg_arr, ecc_perm, ecc_perm_bit_loc, num_meta_cw-1)
    end = time.monotonic()
    elapsed = end-begin
    print(f"AES extraction done in {elapsed} seconds")
    # Decrypt AES
    print("AES decrypting...")
    begin = time.monotonic()
    mesecc_arr = endecrypt_aes(num_ecc_cw, mesecc_arr, aes_key, num_meta_cw-1)
    end = time.monotonic()
    elapsed = end-begin
    print(f"AES decryption done in {elapsed} seconds")
    # Decode ECC
    print("ECC decoding")
    begin = time.monotonic()
    ecc_ok, decmsg_arr = decode_ecc(num_ecc_cw, 2*err_count, mesecc_arr, num_meta_cw-1)
    end = time.monotonic()
    elapsed = end-begin
    print(f"ECC decoding done in {elapsed} seconds")
    if not ecc_ok:
        raise Exception("ECC failed")
    # Done
    return decmsg_arr.tobytes()[combined_meta_len:combined_meta_len+filesize]

# Usage
def print_usage():
    print("Read ISO100 raws from img/")
    print("Usage:")
    print("iso = 200, 400, 800")
    print("Keygen:")
    print(f"\t{sys.argv[0]} k")
    print("Embedding:")
    print(f"\t{sys.argv[0]} e iso msg_file")
    print("Embedding with verification:")
    print(f"\t{sys.argv[0]} e iso msg_file v")
    print("Extraction:")
    print(f"\t{sys.argv[0]} d iso")
    print("Extraction with verification:")
    print(f"\t{sys.argv[0]} d iso original_msg_file")
    sys.exit(1)

# Main function
if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 2:
        print_usage()

    # range for searching optimal embedding
    max_ell = 8
    max_beta = 16

    # parameters for master chunk (Kyber & metadata)
    master_height = 92
    master_width = 96
    master_size = master_height*master_width
    master_dup = 5
    metadata_offset = 8704 # kyber ctxt size = 1088 * 8 bits
    num_pixel_master = master_size*master_dup

    if sys.argv[1] == 'k':
        print("En/decap key generation")
        ek, dk = ML_KEM_768.keygen()  # encapsulation and decapsulation key
        try:
            with open("decap_key.bin", "wb") as f:
                f.write(dk)
            with open("encap_key.bin", "wb") as f:
                f.write(ek)
        except Exception as e:
            print("Can't write key files")
            sys.exit(1)
        print("Done")
        sys.exit(0)

    elif sys.argv[1] == 'e':
        if argc < 4:
            print_usage()
        target_iso = int(sys.argv[2])
        if target_iso != 200 and target_iso != 400 and target_iso != 800:
            print("ISO data not available")
            sys.exit(1)
        try:
            with open("encap_key.bin", "rb") as f:
                ek = f.read()
        except Exception as e:
            print("Can't read encap key file")
            sys.exit(1)

        try:
            with open(sys.argv[3], "rb") as f:
                ori_msg = f.read()
        except Exception as e:
            print("Can't read message file")
            sys.exit(1)
        filesize = len(ori_msg)
        print(f"Message file {sys.argv[3]} size = {filesize}")

        need_verify = argc > 4 and sys.argv[4] == 'v'
        if need_verify:
            try:
                with open("decap_key.bin", "rb") as f:
                    dk = f.read()
            except Exception as e:
                print("Can't read decap key file")
                sys.exit(1)

        if not os.path.exists("./img100/"):
            os.makedirs("./img100/")            
        if not os.path.exists("./img"+sys.argv[2]+"/"):
            os.makedirs("./img"+sys.argv[2]+"/")

        path = "./img/"
        rawls = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".arw")]
        for rawfile in rawls:
            rawpath = os.path.join(path, rawfile)
            iso = piexif.load(rawpath)["Exif"][piexif.ExifIFD.ISOSpeedRatings]
            if iso != 100:
                continue
            print(f"========================================\nProcessing {rawpath}")
            try:
                with rp.imread(rawpath) as f:
                    s = cal_s(f, target_iso)
                    s = s.clip(0)
                    raw = f.raw_image.copy()
            except Exception as e:
                print(f"Can't open {rawpath}")
                continue
            # Save DNG for comparison
            save_dng(raw, "./img100/", rawfile)
            # Embed
            try:
                begin = time.monotonic()
                stego_raw = stego_embedding(raw, s)
                end = time.monotonic()
                elapsed = end-begin
                print(f"Stego embedding done in {elapsed} seconds")
            except Exception as e:
                print(f"{str(e)}")
                continue
            # Save stego DNG
            can_save = True
            if need_verify:
                print("Verifying...")
                try:
                    msg_ = stego_extract(stego_raw)
                except Exception as e:
                    print(f"{str(e)}")
                    continue
                print(f"Verification: {'OK' if msg_ == ori_msg else 'Failed'}")
                if msg_ != ori_msg:
                    can_save = False
            if can_save:
                save_dng(stego_raw, "./img"+sys.argv[2]+"/", rawfile)
                print("Stego saved")


    elif sys.argv[1] == 'd':
        if argc < 3:
            print_usage()
        target_iso = int(sys.argv[2])
        try:
            with open("decap_key.bin", "rb") as f:
                dk = f.read()
        except Exception as e:
            print("Can't read decap key file")
            sys.exit(1)
        
        if not os.path.exists("./dec"+sys.argv[2]+"/"):
            os.makedirs("./dec"+sys.argv[2]+"/")

        path = "./img"+sys.argv[2]+"/"
        rawls = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".dng")]
        for rawfile in rawls:
            rawpath = os.path.join(path, rawfile)
            print(f"========================================\nProcessing {rawpath}")
            try:
                with rp.imread(rawpath) as f:
                    raw = f.raw_image.copy()
            except Exception as e:
                print(f"Can't open {rawpath}")
                continue
            
            try:
                begin = time.monotonic()
                msg = stego_extract(raw)
                end = time.monotonic()
                elapsed = end-begin
                print(f"Stego extraction done in {elapsed} seconds")
            except Exception as e:
                print(f"{str(e)}")
                continue

            savepath = os.path.join("./dec"+sys.argv[2]+"/", rawfile+".bin")
            try:
                with open(savepath, "wb") as f:
                    f.write(msg)
            except Exception as e:
                print("Can't write decoded file")
                continue
            print("Extracted covertext saved")

            if argc > 3:
                print("Verifying...")
                status = subprocess.run(["cmp", savepath, sys.argv[3]]).returncode
                status = status if status < 255 else status%255
                print(f"cmp result: {'OK' if status == 0 else 'Failed'}")
