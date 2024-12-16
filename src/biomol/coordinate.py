import math

import numpy as np
from Bio.PDB import Residue
from typing_extensions import TypedDict

from biomol import rotamer
from biomol.topology import AmberParameter, AmberTopology
from biomol.utils import geometry


# TODO: consider using NamedTuple?
class InternalCoordDict(TypedDict):
    this_atom_idx: int
    bond_atom_idx: int
    bond_distance: float
    angle_atom_idx: int
    angle: float
    dihedral_atom_idx: int
    dihedral: float


def copy_backbone_coordinate(*, residue_obj: Residue.Residue) -> dict[str, np.ndarray]:
    """
    Get Cartesian coordinates of backbone atoms.

    Args:
        residue_obj: Target residue object

    Returns:
        Atom names to coordinate.
    """
    backbone_atom_names = ["N", "H", "CA", "HA", "C", "O"]
    bb_crd: dict[str, np.ndarray] = {}
    for atom_obj in residue_obj:
        if atom_obj.get_name() in backbone_atom_names:
            bb_crd[atom_obj.get_name()] = atom_obj.get_coord()

    return bb_crd


def convert_internal_to_cartesian_coordinate(
    *,
    resname: str,
    zmatrix: dict[str, InternalCoordDict],
    backbone_crd: dict[str, np.ndarray],
    forcefield_topology: AmberTopology,
) -> dict[str, np.ndarray]:
    """
    Convert Internal coordinate (zmatrix) to Cartesian coordinate.

    Args:
    - resname: target residue name, required for AA_ATOM_NAMES, AA_CONNECT
    - zmatrix: (initially) constructed zmatrix
    - backbone_crd: Cartesian coordinates for backbone atoms

    Returns:
        atom_name to x,y,z coordinate

    Reference:
        - xyzatm.f90 in tinker source codes
    """
    eps: float = 0.00000001
    cartcrd: dict[str, np.ndarray] = {}
    ff_top: AmberTopology = forcefield_topology
    res_atom_names: list[str] = ff_top.AA_ATOM_NAMES[resname]

    for atom_name in res_atom_names:
        cartcrd[atom_name] = np.array((0.0, 0.0, 0.0))

    for atom_name, intcrd in zmatrix.items():
        iatm_id = intcrd["this_atom_idx"]
        # bond
        batm_id = intcrd["bond_atom_idx"]
        batm_name = res_atom_names[batm_id - 1]
        bond = intcrd["bond_distance"]
        # angle
        aatm_id = intcrd["angle_atom_idx"]
        aatm_name = res_atom_names[aatm_id - 1]
        angle = intcrd["angle"]
        # dihedral
        datm_id = intcrd["dihedral_atom_idx"]
        datm_name = res_atom_names[datm_id - 1]
        dihed = intcrd["dihedral"]

        sin_angle_rad = 0.0
        cos_angle_rad = 0.0
        sin_dihed_rad = 0.0
        cos_dihed_rad = 0.0
        if not np.isnan(angle):
            angle_rad = math.radians(angle)
            sin_angle_rad = math.sin(angle_rad)
            cos_angle_rad = math.cos(angle_rad)
        if not np.isnan(dihed):
            dihed_rad = math.radians(dihed)
            sin_dihed_rad = math.sin(dihed_rad)
            cos_dihed_rad = math.cos(dihed_rad)

        if batm_id == -1:
            # first atom in zmatrix
            continue
        elif aatm_id == -1:
            # second atom in zmatrix
            cartcrd[atom_name] = cartcrd[batm_name] + np.array((0.0, 0.0, bond))
            continue
        elif datm_id == -1:
            # third atom in zmatrix
            bond_crd = cartcrd[batm_name]  # bond atom's crd
            angl_crd = cartcrd[aatm_name]  # angle atom's crd
            vec = bond_crd - angl_crd  # vector of bond_atom to angle_atom
            norm = np.linalg.norm(vec)  # norm of vector of bond_atom to angle_atom
            unit_vec = vec / norm

            cosb = unit_vec[2]
            sinb = np.sqrt(unit_vec[0] ** 2, unit_vec[1] ** 2)
            if sinb == 0.0:
                cosg = 1.0
                sing = 0.0
            else:
                cosg = unit_vec[1] / sinb
                sing = unit_vec[0] / sinb
            xtmp = bond * sin_angle_rad
            ztmp = norm - bond * cos_angle_rad
            cartcrd[atom_name][0] = bond_crd[0] + xtmp * cosg + ztmp * sing * sinb
            cartcrd[atom_name][1] = bond_crd[1] - xtmp * sing + ztmp * cosg * sinb
            cartcrd[atom_name][2] = bond_crd[2] + ztmp * cosb
        else:
            # fourth atom ~
            bond_crd = cartcrd[batm_name]
            angl_crd = cartcrd[aatm_name]
            dihe_crd = cartcrd[datm_name]
            # vector_ba: bont_atom - angle_atom
            vec_ba = bond_crd - angl_crd
            norm_ba = np.linalg.norm(vec_ba)
            unit_vec_ba = vec_ba / norm_ba
            # vector_ad: angle_atom - dihedral_atom
            vec_ad = angl_crd - dihe_crd
            norm_ad = np.linalg.norm(vec_ad)
            unit_vec_ad = vec_ad / norm_ad
            # vector_ba X vector_ad
            vec_t = np.array((0.0, 0.0, 0.0))
            vec_t[0] = unit_vec_ba[2] * unit_vec_ad[1] - unit_vec_ba[1] * unit_vec_ad[2]
            vec_t[1] = unit_vec_ba[0] * unit_vec_ad[2] - unit_vec_ba[2] * unit_vec_ad[0]
            vec_t[2] = unit_vec_ba[1] * unit_vec_ad[0] - unit_vec_ba[0] * unit_vec_ad[1]
            cosine = (
                unit_vec_ba[0] * unit_vec_ad[0]
                + unit_vec_ba[1] * unit_vec_ad[1]
                + unit_vec_ba[2] * unit_vec_ad[2]
            )
            sine = np.sqrt(np.max([1.0 - cosine**2, eps]))
            if abs(cosine) >= 1.0:
                raise RuntimeError(
                    f"Convert Int2Cart: Undefined Dihedral Angle at atom-{iatm_id} {atom_name}"
                )
            vec_t = vec_t / sine
            vec_u = np.array((0.0, 0.0, 0.0))
            vec_u[0] = vec_t[1] * unit_vec_ba[2] - vec_t[2] * unit_vec_ba[1]
            vec_u[1] = vec_t[2] * unit_vec_ba[0] - vec_t[0] * unit_vec_ba[2]
            vec_u[2] = vec_t[0] * unit_vec_ba[1] - vec_t[1] * unit_vec_ba[0]
            # Calculate coordinates
            cartcrd[atom_name][0] = bond_crd[0] + bond * (
                vec_u[0] * sin_angle_rad * cos_dihed_rad
                + vec_t[0] * sin_angle_rad * sin_dihed_rad
                - unit_vec_ba[0] * cos_angle_rad
            )
            cartcrd[atom_name][1] = bond_crd[1] + bond * (
                vec_u[1] * sin_angle_rad * cos_dihed_rad
                + vec_t[1] * sin_angle_rad * sin_dihed_rad
                - unit_vec_ba[1] * cos_angle_rad
            )
            cartcrd[atom_name][2] = bond_crd[2] + bond * (
                vec_u[2] * sin_angle_rad * cos_dihed_rad
                + vec_t[2] * sin_angle_rad * sin_dihed_rad
                - unit_vec_ba[2] * cos_angle_rad
            )
    # translate and rotate converted Cartesian coordinates
    # to the original backbone coordinates
    cartcrd = transform_coord(reference_crd=backbone_crd, target_crd=cartcrd)

    return cartcrd


def transform_coord(
    *,
    reference_crd: dict[str, np.ndarray],
    target_crd: dict[str, np.ndarray],
    target_atoms: list[str] = ["N", "CA", "C", "O"],
) -> dict[str, np.ndarray]:
    """
    Transforms the target coordinates to the reference coordinates.

    Args:
    reference_crd: coordinates of reference atoms.
    target_crd: coordinates of transforming target atoms.
    target_atoms: atom names for referencing target_crd to the reference_crd.

    Returns:
    transformed_crd: transformed target_crd
    """
    transformed_crd: dict[str, np.ndarray] = {}

    # Reorder atoms' coordinates
    ref_crds = []
    tgt_crds = []
    # first, add atoms' coordinates in the target_atoms
    for atmnam in target_atoms:
        ref_crds.append(reference_crd[atmnam])
        tgt_crds.append(target_crd[atmnam])
    # second, add other atoms' coordinates of target_crd not in the target_atoms
    for atmnam in target_crd:
        if atmnam not in target_atoms:
            tgt_crds.append(target_crd[atmnam])

    # Convert list[np.ndarray] to np.ndarray
    ref_crds = np.array(ref_crds)
    tgt_crds = np.array(tgt_crds)

    # Calculate Kabsch algorithm.
    trans_tgt_crds = geometry.transform_by_kabsch(
        P=np.array(ref_crds), Q=tgt_crds[: len(target_atoms)]
    )

    # Transform the target coordinates
    idx = 0
    for idx, atmnam in enumerate(target_crd.keys()):
        transformed_crd[atmnam] = trans_tgt_crds[idx]

    return transformed_crd


def construct_residue_internal_coord_of_sidechain(
    *,
    resname: str,
    forcefield_topology: AmberTopology,
    forcefield_parameter: AmberParameter,
    chi_angles: list[float] | None = None,
) -> dict[str, InternalCoordDict]:
    """
    Construction of internal coordinates of a residue based on Amber force field.

    Args:
    - resname: target residue name
    - forcefield_topology: amino acid toplogy defined in Amber force field
    - forcefield_parameter: force field energy parameter
    - chi_angles: major side chain dihedral angles (chi 1~4) extracted from rotamer
    """
    ff_top: AmberTopology = forcefield_topology
    res_atom_names: list[str] = ff_top.AA_ATOM_NAMES[resname]
    res_atom_types: dict[str, str] = ff_top.AA_ATOM_TYPES[resname]
    res_atoms_connect: list[tuple[int, int]] = ff_top.AA_CONNECT[resname]
    zmatrix: dict[str, InternalCoordDict]

    zmatrix = build_connectivity_for_internal_coordinate(
        resname=resname, atoms_list=res_atom_names, atoms_connect=res_atoms_connect
    )

    zmatrix = set_geometric_values_from_parameter(
        zmatrix=zmatrix,
        resname=resname,
        forcefield_parameter=forcefield_parameter,
        chi_angles=chi_angles,
        atom_name_list=res_atom_names,
        atom_name_type=res_atom_types,
    )

    return zmatrix


def set_geometric_values_from_parameter(
    *,
    zmatrix: dict[str, InternalCoordDict],
    resname: str,
    forcefield_parameter: AmberParameter,
    chi_angles: list[float] | None = None,
    atom_name_list: list[str],
    atom_name_type: dict[str, str],
    dihedral_angle_idx: int = 0,
) -> dict[str, InternalCoordDict]:
    """
    Setup dihedral angles for side chains.

    Args:
    - zmatrix: internal coordinates
    - forcefield_parameter: geometric values from force field energy parameter
    - chi_angles: explicitly given chi angles (from rotamer library)
    - atom_name_list: list of atom names for indexing atoms in zmatrix
    - atom_name_type: map of atom_name to atom_type
    """
    # update geometric values based on force field parameters
    idx = 1
    for atomname in zmatrix:
        atomtype = atom_name_type[atomname]
        bond_atm_idx = zmatrix[atomname]["bond_atom_idx"]
        angl_atm_idx = zmatrix[atomname]["angle_atom_idx"]
        dihe_atm_idx = zmatrix[atomname]["dihedral_atom_idx"]

        # skip idx == 1
        if idx == 2:  # noqa: PLR2004
            bond_atm_name = atom_name_list[bond_atm_idx - 1]
            bond_atm_type = atom_name_type[bond_atm_name]
            zmatrix[atomname]["bond_distance"] = forcefield_parameter.get_bond_dist(
                atom_i=atomtype, atom_j=bond_atm_type
            )

        elif idx == 3:  # noqa: PLR2004
            bond_atm_name = atom_name_list[bond_atm_idx - 1]
            bond_atm_type = atom_name_type[bond_atm_name]
            zmatrix[atomname]["bond_distance"] = forcefield_parameter.get_bond_dist(
                atom_i=atomtype, atom_j=bond_atm_type
            )
            angl_atm_name = atom_name_list[angl_atm_idx - 1]
            angl_atm_type = atom_name_type[angl_atm_name]
            zmatrix[atomname]["angle"] = forcefield_parameter.get_angle_degree(
                atom_i=atomtype, atom_j=bond_atm_type, atom_k=angl_atm_type
            )

        elif idx >= 4:  # noqa: PLR2004
            bond_atm_name = atom_name_list[bond_atm_idx - 1]
            bond_atm_type = atom_name_type[bond_atm_name]
            zmatrix[atomname]["bond_distance"] = forcefield_parameter.get_bond_dist(
                atom_i=atomtype, atom_j=bond_atm_type
            )
            angl_atm_name = atom_name_list[angl_atm_idx - 1]
            angl_atm_type = atom_name_type[angl_atm_name]
            zmatrix[atomname]["angle"] = forcefield_parameter.get_angle_degree(
                atom_i=atomtype, atom_j=bond_atm_type, atom_k=angl_atm_type
            )
            dihe_atm_name = atom_name_list[dihe_atm_idx - 1]
            dihe_atm_type = atom_name_type[dihe_atm_name]
            zmatrix[atomname]["dihedral"] = forcefield_parameter.get_dihedral_angle(
                atom_i=atomtype,
                atom_j=bond_atm_type,
                atom_k=angl_atm_type,
                atom_l=dihe_atm_type,
                dihed_idx=dihedral_angle_idx,
            )

        else:
            pass

        idx += 1

    # update chi angles using the given rotamer chi angles
    if chi_angles is None:
        chi_angles = []
    chi_names = ["CHI1", "CHI2", "CHI3", "CHI4"]
    for idx, chi_value in enumerate(chi_angles):
        chi = chi_names[idx]
        if resname in rotamer.DIHEDRAL_MAP[chi]:
            this_atom_names = rotamer.DIHEDRAL_MAP[chi][resname]
            # atm_i_type = atom_name_type[this_atom_names[0]]
            # atm_j_type = atom_name_type[this_atom_names[1]]
            # atm_k_type = atom_name_type[this_atom_names[2]]
            # atm_l_type = atom_name_type[this_atom_names[3]]
            zmatrix[this_atom_names[0]]["dihedral"] = chi_value

    return zmatrix


def build_connectivity_for_internal_coordinate(
    *, resname: str, atoms_list: list[str], atoms_connect: list[tuple[int, int]]
) -> dict[str, InternalCoordDict]:
    """
    Build atom connectivity in internal coordinates based on Amber forcefield library and rotamer dihedral map.

    Args:
    - resname: residue name
    - atoms_list: atom name list of a residue
    - atoms_connect: list of atom id pairs (tuple: (i_atom, j_atom))

    Returns:
    - zmatrix: synonym of internal coordinate.
      its format -> this_atom_id bond_atm_id distance angle_atm_id angle dihedral_atm_id dihedral_angle

    (Example)

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
                zmatrix[iatm_name] = InternalCoordDict(
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
                zmatrix[jatm_name] = InternalCoordDict(
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
                zmatrix[atoms_list[jatm - 1]] = InternalCoordDict(
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

                zmatrix[atoms_list[jatm - 1]] = InternalCoordDict(
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
            zmatrix[jatm_name] = InternalCoordDict(
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
