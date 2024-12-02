import os
from pathlib import Path

import pandas as pd
from Bio.PDB import Atom, Residue
from Bio.PDB.vectors import calc_angle, calc_dihedral

rotamer_libfile = f'{os.environ["HOME"]}/rotamerBB/Everything-5_rotamerBB/ExtendedOpt2-5/ALL.bbdep.rotamers.lib'

# NOTE: Explanation of dihedral map structures and terms
#       "dihedral angle name" : {"residue name" : ["atom_name-1","atom_name-2","atom_name-3","atom_name-4"]}
#       "ANY" : any amino acids
#       "NTANY" : N-terminal amino acid
#       "CTANY" : C-terminal amino acid
#       "CY=" : Cysteine amino acid names. e.g. CYS, CYM (Metal chelating), CYX (disulfide)
#       "HI=" : Histidine amino acid names. e.g. HIS, HID, HIE, HIP (~D,~E,~P : Amber type Histidine)
#       "-C" : backbone C atom in previous amino acid
#       "+N" : backbone N atom in next amino acid
#       "+CA" : backbone C-alpha atom in next amino acid
#       "XD" : delta position atom in side chain
#       "XD1" : delta1 position atom in side chain
#       "XD2" : detal2 position atom in side chain
# NOTE: Reference of definition of side-chain dihedral angles
#       https://dunbrack.fccc.edu/bbdep2010/ConformationalAnalysis.php
#       https://salilab.org/pdf/Sali_ProtStrucDistAna_1994.pdf
#          -> "Comparative Protein Modeling by Satisfaction of Spatial Restraints

DIHEDRAL_MAP = {
    "PHI": {"ANY": ["-C", "N", "CA", "C"], "NTANY": ["H", "N", "CA", "C"]},
    "PSI": {"ANY": ["N", "CA", "C", "+N"], "CTANY": ["N", "CA", "C", "OXT"]},
    "OMEGA": {"ANY": ["CA", "C", "+N", "+CA"]},
    "CHI1": {
        "VAL": ["N", "CA", "CB", "CG1"],
        "ILE": ["N", "CA", "CB", "CG1"],
        "SER": ["N", "CA", "CB", "OG"],
        "THR": ["N", "CA", "CB", "OG1"],
        "CY=": ["N", "CA", "CB", "SG"],
        "LEU": ["N", "CA", "CB", "CG"],
        "MET": ["N", "CA", "CB", "CG"],
        "PHE": ["N", "CA", "CB", "CG"],
        "TRP": ["N", "CA", "CB", "CG"],
        "HI=": ["N", "CA", "CB", "CG"],
        "LYS": ["N", "CA", "CB", "CG"],
        "ARG": ["N", "CA", "CB", "CG"],
        "ASP": ["N", "CA", "CB", "CG"],
        "GLU": ["N", "CA", "CB", "CG"],
        "TYR": ["N", "CA", "CB", "CG"],
        "ASN": ["N", "CA", "CB", "CG"],
        "GLN": ["N", "CA", "CB", "CG"],
    },
    "CHI2": {
        "LEU": ["CA", "CB", "CG", "CD1"],
        "ARG": ["CA", "CB", "CG", "XD"],
        "LYS": ["CA", "CB", "CG", "XD"],
        "MET": ["CA", "CB", "CG", "XD"],
        "GLN": ["CA", "CB", "CG", "XD"],
        "GLU": ["CA", "CB", "CG", "XD"],
        "ILE": ["CA", "CB", "CG", "XD"],
        "PHE": ["CA", "CB", "CG", "XD1"],
        "TYR": ["CA", "CB", "CG", "XD1"],
        "HI=": ["CA", "CB", "CG", "XD1"],
        "TRP": ["CA", "CB", "CG", "XD1"],
        "ASP": ["CA", "CB", "CG", "XD1"],
        "ASN": ["CA", "CB", "CG", "XD2"],
    },
    "CHI3": {
        "LYS": ["CB", "CG", "CD", "CE"],
        "ARG": ["CB", "CG", "CD", "NE"],
        "GLU": ["CB", "CG", "CD", "OE1"],
        "GLN": ["CB", "CG", "CD", "OE1"],
        "MET": ["CB", "CG", "SD", "CE"],
    },
    "CHI4": {
        "LYS": ["CG", "CD", "CE", "NZ"],
        "ARG": ["CG", "CD", "NE", "CZ"],
    },
}


def load_rotamer_library() -> pd.DataFrame:
    assert Path(
        rotamer_libfile
    ).exists(), f"FileNotFoundError: {rotamer_libfile} does not exist!!"

    header_type = {
        "residue": str,
        "Phi": int,
        "Psi": int,
        "Count": int,
        "r1": int,
        "r2": int,
        "r3": int,
        "r4": int,
        "Probabil": float,
        "chi1Val": float,
        "chi2Val": float,
        "chi3Val": float,
        "chi4Val": float,
        "chi1Sig": float,
        "chi2Sig": float,
        "chi3Sig": float,
        "chi4Sig": float,
    }
    rotamer_library = pd.read_csv(
        rotamer_libfile,
        delim_whitespace=True,
        comment="#",
        names=list(header_type.keys()),
        # dtype=header_type,
        header=None,
    )
    # print(rotamer_library.dtypes)
    return rotamer_library


def get_sidechain_rotamers(
    rotamer_lib: pd.DataFrame,
    resname: str,
    phi: float,
    psi: float,
    sort_by: str = "Probabil",
    rank: int = 0,
) -> list[float]:
    """
    Args.

    sort_by : column for sorting selected dataframe
    rank : rank of data selected after sorted by [sort_by] column, 0: first
    """
    rotamers = [0.0, 0.0, 0.0, 0.0]
    phi_key = round(phi, -1)
    psi_key = round(psi, -1)
    possible_rotamers: pd.DataFrame = rotamer_lib[  # type: ignore
        (rotamer_lib["Residue"] == resname)
        & (rotamer_lib["Phi"] == phi_key)
        & (rotamer_lib["Psi"] == psi_key)
    ]
    sort_possible_rotamers = possible_rotamers.sort_values(
        by=[sort_by], ascending=False
    )
    rotamers = list(
        sort_possible_rotamers.iloc[rank][["chi1Val", "chi2Val", "chi3Val", "chi4Val"]]
    )

    return rotamers


def get_atom_with_specific_position_in_sidechain(
    residue_obj: Residue.Residue, position_key: str
):
    atom_obj = Atom.Atom
    find_atom = False
    searchable_atom_name_minimum_length = 2
    single_pos = 1
    double_pos = 2
    for atom in residue_obj.get_atoms():
        if len(atom.get_name()) >= searchable_atom_name_minimum_length:
            """Atom name format: 'XYZ'
            X: Atom type (C, N, O, S)
            Y: Position (alpha: A, beta: B, gamma: G, delta: D, epsilon: E, zeta: Z
            Z: Alternative position id (e.g. CD1 & CD2, CE1 & NE2)
            """
            if (
                len(position_key) == single_pos and position_key == atom.get_name()[1]
            ) or (
                len(position_key) == double_pos and position_key == atom.get_name()[1:]
            ):
                atom_obj = atom
                find_atom = True
                break
    if not find_atom:
        return None
    else:
        return atom_obj


def get_dihedral_angle_atom_composition(
    dihedral_angle_type: str,
    residue_name: str,
    *,
    n_terminal=False,
    c_terminal=False,
):
    assert not (n_terminal and c_terminal), "Amino acid can not be N-term and C-term"
    dihedral_angle_type = dihedral_angle_type.upper()
    residue_name = residue_name.upper()

    if dihedral_angle_type in ["PHI", "PSI", "OMEGA"]:
        if not n_terminal and not c_terminal:
            atom_list = DIHEDRAL_MAP[dihedral_angle_type]["ANY"]
        elif n_terminal and not c_terminal:
            atom_list = DIHEDRAL_MAP[dihedral_angle_type]["NTANY"]
        else:
            atom_list = DIHEDRAL_MAP[dihedral_angle_type]["CTANY"]
    else:
        if residue_name in ["CYS", "CYX", "CYM"]:
            residue_name = "CY="
        elif residue_name in ["HIS", "HIE", "HID", "HIP"]:
            residue_name = "HI="
        else:
            pass
        try:
            atom_list = DIHEDRAL_MAP[dihedral_angle_type][residue_name]
        except KeyError:
            atom_list = [None, "", "", ""]

    return atom_list


def calc_dihedral_angle(
    residue_obj: Residue.Residue,
    dihedral_angle_type: str,
    residue_name: str,
    prev_residue_obj: Residue.Residue | None = None,
    next_residue_obj: Residue.Residue | None = None,
    *,
    n_terminal=False,
    c_terminal=False,
):
    at1name, at2name, at3name, at4name = get_dihedral_angle_atom_composition(
        dihedral_angle_type, residue_name, n_terminal=n_terminal, c_terminal=c_terminal
    )
    assert isinstance(
        at1name, str
    ), f"Failed to find atoms for dihedral angles of {dihedral_angle_type.upper()}"

    """ Atom-1's coordinate vector """
    if dihedral_angle_type.upper() == "PHI" and not n_terminal:
        assert isinstance(
            prev_residue_obj, Residue.Residue
        ), "For Phi angle in non-N-terminal, previous residue is required."
        at1vec = prev_residue_obj["C"].get_vector()
    else:
        at1vec = residue_obj[at1name].get_vector()

    """ Atom-2's coordinate vector """
    at2vec = residue_obj[at2name].get_vector()

    """ Atom-3's coordinate vector """
    if dihedral_angle_type.upper() == "OMEGA":
        assert isinstance(
            next_residue_obj, Residue.Residue
        ), "For Omega angle, next residue should be required."
        at3vec = next_residue_obj["N"].get_vector()
    else:
        at3vec = residue_obj[at3name].get_vector()

    """ Atom-4's coordinate vector """
    if dihedral_angle_type.upper() == "PSI" and not c_terminal:
        assert isinstance(
            next_residue_obj, Residue.Residue
        ), "For Psi angle in Non-C-terminal, next residue is required."
        at4vec = next_residue_obj["N"].get_vector()
    elif dihedral_angle_type.upper() == "OMEGA":
        assert isinstance(
            next_residue_obj, Residue.Residue
        ), "For Omega angle, next residue should be required."
        at4vec = next_residue_obj["CA"].get_vector()
    elif "X" in at4name:
        atom_position_key = at4name[1:]
        at4obj = get_atom_with_specific_position_in_sidechain(
            residue_obj, atom_position_key
        )
        assert isinstance(
            at4obj, Atom.Atom
        ), f"There is no atom of position {atom_position_key} in residue {residue_obj.get_id()}"
        at4vec = at4obj.get_vector()
    else:
        at4vec = residue_obj[at4name].get_vector()

    dihedral = calc_dihedral(at1vec, at2vec, at3vec, at4vec)
    return dihedral


def calc_three_atoms_angle(
    atom1: Atom.Atom, atom2: Atom.Atom, atom3: Atom.Atom
) -> float:
    at1vec = atom1.get_vector()
    at2vec = atom2.get_vector()
    at3vec = atom3.get_vector()

    three_atoms_angle = calc_angle(at1vec, at2vec, at3vec)

    return three_atoms_angle


def calc_bond_distance(atom1: Atom.Atom, atom2: Atom.Atom) -> float:
    bond_dist = atom1 - atom2

    return bond_dist


if __name__ == "__main__":
    load_rotamer_library()
