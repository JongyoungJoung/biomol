from typing import TypedDict

import numpy as np

from biomol.rotamer import DIHEDRAL_MAP
from biomol.topology import AmberParameter, AmberTopology


class IntCoord(TypedDict):
    this_atom_idx: int
    bond_atom_idx: int
    bond_distance: float
    angle_atom_idx: int
    angle: float
    dihedral_atom_idx: int
    dihedral: float


def construct_residue_internal_coord_of_sidechain_from_Amber_topology(  # noqa: N802
    *,
    resname: str,
    forcefield_topology: AmberTopology,
    forcefield_parameter: AmberParameter,
) -> dict[str, IntCoord]:
    ff_top = forcefield_topology
    res_atoms = ff_top.AA_ATOMS[resname]
    res_atoms_connect = ff_top.AA_CONNECT[resname]

    zmatrix = build_connectivity_for_internal_coordinate(
        resname=resname, atoms_list=res_atoms, atoms_connect=res_atoms_connect
    )
    zmatrix = initialize_geometric_values(zmatrix)

    return zmatrix


def initialize_geometric_values(zmatrix) -> dict[str, IntCoord]:
    return zmatrix


def build_connectivity_for_internal_coordinate(
    *, resname: str, atoms_list: list[str], atoms_connect: list[tuple[int, int]]
) -> dict[str, IntCoord]:
    """
    Build atom connectivity in internal coordinates based on Amber forcefield library and rotamer dihedral map.

    # Input
    atoms_list : atom name list of a residue
    atoms_connect : list of atom id pairs (tuple: (i_atom, j_atom))

    # Output
    zmatrix: synonym of internal coordinate.
    format: this_atom_id bond_atm_id distance angle_atm_id angle dihedral_atm_id dihedral_angle
    example:

     (topology)   (Amber forcefield order)  (Amber forcefield connectivity)

            HB2             "N"   : 1                       1 2
             |              "H"   : 2                       1 3
         HB1-CB-HB3         "CA"  : 3                       3 4
             |              "HA"  : 4                       3 5
    (-C)~~N-CA-C=O          "CB"  : 5                       3 9
          |  |              "HB1" : 6                       5 6
          H  HA             "HB2" : 7                       5 7
                            "HB3" : 8                       5 8
                            "C"   : 9                       9 10
                            "O"   : 10
    ZMatrix
    #atm  idx batm_id dist   aatm_id   angle   datm_id     dihedral_angle
    "N"     1   None    -       None    -       None        -
    "CA"    3   1       2.5     None    -       None        -
    "CB"    5   3       2.5     1       120.0   None        -
    "C"     9   3       2.6     1       120.0   (-C)'s_id   180.0
    "O"     10  9       2.3     3       120.0   1           -180.0
    "H"     2   1       1.5     3       120.0   9           0.0         # backbone H
    "HA"    4   3       1.5     9       120.0   10          180.0       # backbone H
    "HB1"   6   5       1.5     3       120.0   1           60.0
    "HB2"   7   5       1.5     3       120.0   1           120.0
    "HB3"   8   5       1.5     3       120.0   1           180.0

    # implicit rule for atom connections for dihedral (4-atom) and angle (3-atom)
    1. Backbone atoms (N or C) assinged as backbone H's dihedral atom
    2. For side-chain H atoms, angle and dihedral atoms should be the earlycommer atoms in library
    3. If bond_atom_idx, angle_atom_idx and dihedral_atom_idx are assigned as -1,
       skip the calculation of geometric values.
    4. In this
    """
    zmatrix = {}
    for iatm, jatm in atoms_connect:
        # iatm : always heavy atom. This shouldn't be "Hxx" atom
        # jatm : heavy atom or H atom
        iatm_name = atoms_list[iatm - 1]
        jatm_name = atoms_list[jatm - 1]
        # for indexing, jatm-1
        # In the first loop, only check the second atoms' type in atom_connect
        if jatm_name[0] != "H":
            # for non-hydrogen atoms
            if len(zmatrix) == 0:
                # first atom : N (backbone)
                zmatrix[iatm_name] = IntCoord(
                    {
                        "this_atom_idx": iatm,
                        "bond_atom_idx": -1,
                        "bond_distance": np.nan,
                        "angle_atom_idx": -1,
                        "angle": np.nan,
                        "dihedral_atom_idx": -1,
                        "dihedral": np.nan,
                    }
                )
                # second atom CA (backbone)
                zmatrix[jatm_name] = IntCoord(
                    {
                        "this_atom_idx": jatm,
                        "bond_atom_idx": iatm,
                        "bond_distance": np.nan,
                        "angle_atom_idx": -1,
                        "angle": np.nan,
                        "dihedral_atom_idx": -1,
                        "dihedral": np.nan,
                    }
                )
            elif len(zmatrix) == 2:
                # third atom : CB or C (backbone, GLY)
                angle_atom_idx = zmatrix[iatm_name]["bond_atom_idx"]
                zmatrix[atoms_list[jatm - 1]] = IntCoord(
                    {
                        "this_atom_idx": jatm,
                        "bond_atom_idx": iatm,
                        "bond_distance": np.nan,
                        "angle_atom_idx": angle_atom_idx,
                        "angle": np.nan,
                        "dihedral_atom_idx": -1,
                        "dihedral": np.nan,
                    }
                )
            elif len(zmatrix) >= 3:
                # fourth atom ~ last atom
                angle_atom_idx = zmatrix[iatm_name]["bond_atom_idx"]
                angle_atom_name = atoms_list[angle_atom_idx - 1]
                dihed_atom_idx = zmatrix[angle_atom_name]["bond_atom_idx"]

                zmatrix[atoms_list[jatm - 1]] = IntCoord(
                    {
                        "this_atom_idx": jatm,
                        "bond_atom_idx": iatm,
                        "bond_distance": np.nan,
                        "angle_atom_idx": angle_atom_idx,
                        "angle": np.nan,
                        "dihedral_atom_idx": dihed_atom_idx,
                        "dihedral": np.nan,
                    }
                )
            else:
                pass
    # In the second loop, only consider when the second atom is Hydrogen
    for iatm, jatm in atoms_connect:
        iatm_name = atoms_list[iatm - 1]
        jatm_name = atoms_list[jatm - 1]
        if jatm_name[0] == "H":
            # H  : H-N-CA-C
            # HA : HA-CA-C-O
            if jatm_name == "H":
                angle_atom_idx = atoms_list.index("N") + 1
                dihed_atom_idx = atoms_list.index("CA") + 1
            elif jatm_name == "HA":
                angle_atom_idx = atoms_list.index("C") + 1
                dihed_atom_idx = atoms_list.index("O") + 1
            else:
                angle_atom_idx = zmatrix[iatm_name]["bond_atom_idx"]
                angle_atom_name = atoms_list[angle_atom_idx - 1]
                dihed_atom_idx = zmatrix[angle_atom_name]["bond_atom_idx"]
            zmatrix[jatm_name] = IntCoord(
                {
                    "this_atom_idx": iatm,
                    "bond_atom_idx": jatm,
                    "bond_distance": np.nan,
                    "angle_atom_idx": angle_atom_idx,
                    "angle": np.nan,
                    "dihedral_atom_idx": dihed_atom_idx,
                    "dihedral": np.nan,
                }
            )

    return zmatrix
