#!/usr/bin/env python

import argparse

from numpy import linalg as npla

from biomol import protein

PEPTIDE_BOND_DISTANCE = 1.3
CA_CA_DISTANCE = 3.8
DIST_BUFF = 0.3


def parse_args():
    parser = argparse.ArgumentParser("Check chain breakage of a protein")
    parser.add_argument(
        "-pdb", required=True, dest="pdbfile", type=str, help="pdb file"
    )
    parser.add_argument(
        "-verbose",
        dest="verbose",
        action="store_true",
        help="Show detail of breakage information",
    )

    args = parser.parse_args()
    return args


def run(args):
    pdbobj = protein.Protein(args.pdbfile)
    residue_ids = pdbobj.get_residue_ids()

    ch_nbreak = {}
    has_breakage = False
    for chid in residue_ids:
        n_break_points = 0
        prev_resid = -999
        break_points = []
        for resid in residue_ids[chid]:
            if prev_resid != -999:
                if prev_resid + 1 != resid:
                    prev_res_obj = pdbobj.get_residue_of_specific_resid(
                        prev_resid, chid
                    )
                    this_res_obj = pdbobj.get_residue_of_specific_resid(resid, chid)

                    check_peptide_link_dist = False

                    # TODO:
                    # LSP cannot detect biopyython attributes used in biomol package
                    try:
                        prev_atom_obj = prev_res_obj["C"]
                        this_atom_obj = this_res_obj["N"]
                        check_peptide_link_dist = True
                    except KeyError:
                        try:
                            prev_atom_obj = prev_res_obj["CA"]
                        except KeyError:
                            prev_atom_obj = list(prev_res_obj.get_atoms())[0]
                        try:
                            this_atom_obj = this_res_obj["CA"]
                        except KeyError:
                            this_atom_obj = list(this_res_obj.get_atoms())[0]
                    # Calculate breakage distance
                    breakage_distance = npla.norm(
                        prev_atom_obj.get_coord() - this_atom_obj.get_coord()
                    )
                    # Check
                    if (
                        check_peptide_link_dist
                        and breakage_distance >= PEPTIDE_BOND_DISTANCE + DIST_BUFF
                    ) or (
                        not check_peptide_link_dist
                        and breakage_distance >= CA_CA_DISTANCE + DIST_BUFF
                    ):
                        break_points.append((prev_resid, resid))
                        n_break_points += 1
                        if not has_breakage:
                            has_breakage = True

            prev_resid = resid
        ch_nbreak[chid] = [n_break_points, break_points]

    if args.verbose:
        if has_breakage:
            print("This protein has Breakage Points in the backbone")
        else:
            print("This protein has NO breakage")
        for chid in ch_nbreak:
            print(f"{chid:2s} chain has {ch_nbreak[chid][0]} breakages")
            print(f"          - breakage points: {ch_nbreak[chid][1]}")
    else:
        print(has_breakage)


if __name__ == "__main__":
    args = parse_args()

    run(args)
