"""
This script converts the genome fasta file to a memory map
file of genome one-hot encodings. This is used to accelerate
genome sequence retrieval (Orca automatically detects the memmap
file and use it) and is required if you use the training scripts.

Example usage: python make_genome_memmap.py <path_to_fasta_file>
"""

import sys
import pathlib
import os
from argparse import ArgumentParser

from selene_utils import MemmapGenome

if __name__ == "__main__":
    parser = ArgumentParser("Creating memmap of the genome")
    parser.add_argument("inp_fasta",
                         help="Path to fasta file.")

    args = parser.parse_args()

    mg = MemmapGenome(
          input_path=os.path.abspath(args.inp_fasta),
          memmapfile=os.path.abspath(args.inp_fasta) + ".mmap",
          init_unpicklable=True,
    )
