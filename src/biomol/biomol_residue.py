STANDARD_AMINO_ACIDS = [
    "GLY",
    "ALA",
    "SER",
    "THR",
    "CYS",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PRO",
    "PHE",
    "TYR",
    "TRP",
    "ASP",
    "GLU",
    "ASN",
    "GLN",
    "HIS",
    "HIE",
    "HID",
    "HIP",
    "LYS",
    "ARG",
    "CYX",
    "CYM",
    "SEP",
    "TPO",
    "ACE",
    "NME",
    "S1P",
    "T1P",
    "Y1P",
    "PTR",
    "H1D",
    "H2D",
    "H1E",
    "H2E",
]

METAL_IONS = ["ZN", "MG", "FE", "CA", "NA", "K", "MN", "CU"]

SOLVENT = ["WAT", "HOH", "Na+", "Cl-"]

COFACTOR = ["GDP", "GTP", "GCP", "GSP", "GNP", "ATP", "ANP", "ACP"]


def is_amino_acid(resname: str):
    return resname in STANDARD_AMINO_ACIDS


def is_solvent(resname: str):
    return resname in SOLVENT
