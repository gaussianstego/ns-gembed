README
=====

This repo provides a python code for the feasibility of G-embedding stego photos. The camera is assumed to be Sony A6400.
Put the raw ISO100 .arw photos in the folder `img`. Only support stego in ISO200, 400 and 800.
Uncompressed DNG files will be created. The ISO100 photos (without embedding) are put in the folder `img100`.
Some reusable information will be precomputed in the first run.

The program was run to embed `test.txt` as a demonstration. The stego photos are in the folders `img200`, `img400` and `img800`. The log during embedding and extracting are recorded in `out*.txt` and `verify*.txt` for reference.

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

