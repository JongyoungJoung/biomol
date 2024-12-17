#!/usr/bin/env python

import argparse

from biomol import protein


def parse_arg():
    parser = argparse.ArgumentParser("Mutate protein residue")
    parser.add_argument(
        "-pdb", required=True, dest="pdbfile", type=str, help="pdb file"
    )
    parser.add_argument(
        "-ch", required=True, dest="chain_id", type=str, help="chain id"
    )
    parser.add_argument(
        "-resid", required=True, dest="resid", type=int, help="residue id"
    )
    parser.add_argument(
        "-mutate-residue-name",
        required=True,
        dest="mut_res",
        type=str,
        help="residue type for mutation",
    )

    args = parser.parse_args()
    return args


def run(args):
    pdbobj = protein.Protein(args.pdbfile)

    pdbobj.mutate(resid=args.resid, chid=args.chain_id, mut_res_name=args.mut_res)


if __name__ == "__main__":
    args = parse_arg()

    run(args)
