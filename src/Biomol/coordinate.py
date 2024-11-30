from typing import TypedDict

from Biomol.topology import AA_ATOMS


class IntCoord(TypedDict):
    bondAtomIdx: int
    bondDistance: float
    angleAtomIdx: int
    angle: float
    dihedAtomIdx: int
    dihedral: float


class AminoAcidTopology:
    def __init__(self, amino_acid_name: str):
        self.amino_acid_name = amino_acid_name

    def construct_internal_coord(self, amino_acid_name: str):
        # Collect atoms of a given amino acid
        self.atom_list = AA_ATOMS[amino_acid_name]
        # internal coordinate
        self.int_crd = {}
