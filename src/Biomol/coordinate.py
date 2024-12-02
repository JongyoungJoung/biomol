from typing import TypedDict

from biomol.topology import AmberTopology

# from Biomol.topology import AA_ATOMS


class IntCoord(TypedDict):
    bondAtomIdx: int
    bondDistance: float
    angleAtomIdx: int
    angle: float
    dihedAtomIdx: int
    dihedral: float


def construct_residue_internal_coord(
    *, resname: str, forcefield_topology: AmberTopology
):
    ff_top = forcefield_topology

    for atom_name in ff_top.AA_ATOMS:
        bond_atom
        angle_atom
        dihed_atom
    return None
