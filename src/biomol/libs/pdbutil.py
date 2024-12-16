from . import pdbutil_fortran


def get_interface_residues(
    chain1_coord: list[float],
    chain2_coord: list[float],
    cut: float,
) -> tuple[list[int], int]:
    return pdbutil_fortran.get_interface_residues(
        chain1_coord, chain2_coord, cut, len(chain1_coord), len(chain2_coord)
    )
