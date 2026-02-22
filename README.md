README
======

This repository provides the Python source code and experiment results reported in "Look Ahead! Practical CCA-secure Steganography: Cover-Source Switching meets Lattice Gaussian Sampling" by Russell W. F. Lai, Ivy K. Y. Woo and Hoover H. F. Yin in EUROCRYPT 2026. In brief, the code and the results demonstrate the feasibility of stego-embedding messages as Gaussian noise in digital photographs using lattice-based preimage sampling techniques.  

The given photographs are assumed to be taken by a Sony A6400 camera at ISO100. The code only supports stego-embedding into ISO200, 400 and 800 photos. 

Put the raw ISO100 .arw photos in the folder `img`. Uncompressed DNG files will be created. The ISO100 photos (without embedding) are put in the folder `img100`. Some reusable information will be precomputed in the first run.

The code was run to embed `test.txt` as a demonstration. The stego photos are in the folders `img200`, `img400` and `img800`. The log during embedding and extracting are recorded in `out*.txt` and `verify*.txt` for reference.

## Usage
`iso = 200, 400, 800`

Keygen:
`stego.py k`

Embedding:
`stego.py e iso msg_file`

Embedding with verification:
`stego.py e iso msg_file v`

Extraction:
`stego.py d iso`

Extraction with verification:
`stego.py d iso original_msg_file`

