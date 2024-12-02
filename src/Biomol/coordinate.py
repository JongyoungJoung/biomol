from typing import TypedDict

import numpy as np

from biomol.topology import AmberTopology


class IntCoord(TypedDict):
    this_atom_idx: int
    bond_atom_idx: int
    bond_distance: float
    angle_atom_idx: int
    angle: float
    dihedral_atom_idx: int
    dihedral: float


def construct_residue_internal_coord_from_Amber_topology(
    *, resname: str, forcefield_topology: AmberTopology
) -> dict[str, IntCoord]:
    ff_top = forcefield_topology
    res_atoms = ff_top.AA_ATOMS[resname]
    res_atoms_connect = ff_top.AA_CONNECT[resname]
    zmatrix = build_connectivity(atoms_list=res_atoms, atoms_connect=res_atoms_connect)

    return zmatrix


def build_connectivity(
    *, atoms_list: list[str], atoms_connect: list[(int, int)]
) -> dict[str, IntCoord]:
    int_crd_count = 1
    # 1. skip H atom, construct the connectivity z-matrix among heavy atoms
    for iatm, jatm in res_atoms_connect:
        # iatm : always heavy atom. This shouldn't be "Hxx" atom
        # jatm : heavy atom or H atom

        zmatrix = {}
        if jatm[0] != "H":
            if len(zmatrix) == 0:
                zmatrix[iatm] = IntCoord(
                    {
                        "this_atom_idx": res_atoms.index(iatm) + 1,
                        "bond_atom_idx": -1,
                        "bond_distance": np.nan,
                        "angle_atom_idx": -1,
                        "angle": np.nan,
                        "dihedral_atom_idx": -1,
                        "dihedral": np.nan,
                    }
                )
                zmatrix[jatm] = IntCoord(
                    {
                        "this_atom_idx": res_atoms.index(jatm) + 1,
                        "bond_atom_idx": iatm,
                        "bond_distance": np.nan,
                        "angle_atom_idx": -1,
                        "angle": np.nan,
                        "dihedral_atom_idx": -1,
                        "dihedral": np.nan,
                    }
                )
        else:
            continue
        int_crd_count += 1
