#!/usr/bin/env python

import argparse
import sys
import time

from tqdm import tqdm

from jounglab import protein
from jounglab.libs import pdbutil


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(
            description = "Get number of residue-residue contacting pairs between two proteins"
            )
    parser.add_argument("-p1", "--protein1", dest="pdbfile1", type=str, required=True,
                    help="pdbfile-1")
    parser.add_argument("-p2", "--protein2", dest="pdbfile2", type=str, required=True,
                    help="pdbfile-2")
    parser.add_argument("-cut", "--distance-cut", dest="distance_cutoff", type=float, default=5.0,
                    help="Contact distance cutoff. default: 5.0 A")
    parser.add_argument("-out-list", "--get-contacting-list", dest="outlist", action="store_true",
                    help="Show residue-residue contact list. default: False (only num_contact)")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                    help="Show the results in more detail. default: False (show results simply")
    # fmt: on
    return parser.parse_args()


def show_residue_residue_contacts(args):
    protein1 = protein.Protein(args.pdbfile1)
    protein2 = protein.Protein(args.pdbfile2)

    pro1crd = protein1.get_crds_list_with_resid()
    pro2crd = protein2.get_crds_list_with_resid()

    pairs, npair = pdbutil.get_interface_residues(
        pro1crd, pro2crd, args.distance_cutoff, len(pro1crd), len(pro2crd)
    )
    if args.verbose:
        print(npair)
    else:
        print(f"Num of residue-residue contact pairs: {npair}")
    if args.outlist:
        if args.verbose:
            print(f"Distance cutoff: {args.distance_cutoff:.1f}")
            print(
                f"Contacted res of chain-1: {' '.join(np.unique(pairs[:npair,0])).astype(str)}"
            )
            print(
                f"Contacted res of chain-2: {' '.join(np.unique(pairs[:npair,1])).astype(str)}"
            )
        else:
            print(" ".join(np.unique(pairs[:npair, 0])).astype(str))
            print(" ".join(np.unique(pairs[:npair, 1])).astype(str))


if __name__ == "__main__":
    args = get_args()
    show_residue_residue_contacts(args)
