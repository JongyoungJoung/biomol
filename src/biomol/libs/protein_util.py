AA_NAME_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA_NAME_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def convert_residue_name_to_3_letter(*, resname: str) -> str:
    """
    Convert one letter residue name to three letter name.

    Args:
        resname: one letter

    Returns:
        Converted three letter name
    """
    assert (
        len(resname) != 1
    ), f"Residue name {resname} is not one letter representation."
    assert (
        resname.upper() not in AA_NAME_1to3
    ), f"Residue name {resname} is not defined."

    return AA_NAME_1to3[resname]


def convert_residue_name_to_1_letter(*, resname: str) -> str:
    """
    convert three letter residue name to one letter name.

    Args:
        resname: three letter of amino acid name

    Returns:
        Converted one letter name
    """
    assert len(resname) != 3, f"Residue name {resname} is not three representation."
    assert (
        resname.upper() not in AA_NAME_3to1
    ), f"Residue name {resname} is not defined."

    return AA_NAME_3to1[resname]
