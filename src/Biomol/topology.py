#
# NOTE: Reference Framework of Protein Amino Acids' Topology
#
#       self.AA_ATOMS: Atom compositions of amino acids
#       reference 1) Amber topologoy data $AMBERHOME/dat/leap/lib/amino19.lib
#       reference 2) Page 27 in "https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf"
#       The order of atoms follows that of amber force field library.
#
#       self.AA_CONNECT: Atom connectivity of amino acids
#       "Residue name" : [(atom_i_idx, atom_j_idx)]
#       atom index starts from 1 (first atom defined in topology dictionary)
#       reference: $AMBERHOME/dat/leap/lib/amino19.lib
#                  "!entry.XXX.unit.connectivity" in the library file
import os
import re

# from pathlib import Path
from typing import Literal, TypedDict

import numpy as np


class AtomParam(TypedDict):
    mass: float
    polarizability: float
    lj_12_power_term: float
    lj_6_power_term: float


class BondParam(TypedDict):
    force_const: float
    equil_bond_length: float


class AngleParam(TypedDict):
    force_const: float
    equil_angle: float


class DihedralParam(TypedDict):
    factor_torsion_barrier_divided: int
    barrier_height: list[float]
    phase_shift_angle: list[float]
    periodicity: list[int]


def get_forcefield_type_related_cmdfile(
    *, fftype: str | os.PathLike, get_libfile: bool = False, get_parmfile: bool = False
) -> list[str]:
    """
    Select Amber forcefield cmd file according to fftype.

    leaprc.[fftype] : cmd file that contains forcefield parameters and libraries.
    """
    cmd_file = f"{os.environ('AMBERHOME')}/dat/leap/cmd/leaprc.{fftype}"
    libfiles: list[str] = []
    parmfiles: list[str] = []

    # extract library files or parameter files matched with fftype
    with open(cmd_file) as cmd:
        for line in cmd:
            if "loadamberparms" in line:
                parmfiles.append(line.split()[-1])
            elif line.startswith("loadOff"):
                libfiles.append(line.split()[-1])
            else:
                pass

    if get_libfile and not get_parmfile:
        return libfiles
    elif not get_libfile and get_parmfile:
        return parmfiles
    else:
        return libfiles


class AmberTopology:
    """
    Class that containes amino acids' topologies defined in Amber forcefield library.

    @ Main contents
        Residue types
        Atoms in the residue type
        Atom connectivities of the residue type
    """

    def __init__(
        self,
        fftype: Literal["ff99SB", "ff12SB", "ff14SB", "ff19SB"] = "ff19SB",
    ):
        assert "AMBERHOME" in os.environ, self.report_error()
        self.AA_ATOMS = {}
        self.AA_CONNECT = {}

        # TODO: change this for alternative options of force field. e.g ff14SB, ff19SB, etc
        self.forcefield_type = fftype
        self.forcefield_libfile = get_forcefield_type_related_cmdfile(
            fftype=fftype, get_libfile=True, get_parmfile=False
        )

        self.parse_ff_library()

    def parse_ff_library(self):
        """
        Parse Amber forcefield library file which defines topology of amino acids.
        """
        new_amino_acid = False
        connectivity = False
        resname = ""
        atomname = ""

        for libfile in self.forcefield_libfile:
            with open(libfile) as library:
                for line in library:
                    if re.match(line.split()[0], r"!entry\.\w{3}\.unit\.atomspertinfo"):
                        new_amino_acid = True
                        resname = line.split()[0].split(".")[1]
                        self.AA_ATOMS[resname] = []
                        continue
                    elif re.match(line.split()[0], r"!entry\.\w{3}\.unit\.boundbox"):
                        new_amino_acid = False
                        continue
                    elif re.match(
                        line.split()[0], r"!entry\.\w{3}\.unit\.connectivity"
                    ):
                        resname = line.split()[0].split(".")[1]
                        self.AA_CONNECT[resname] = []
                        connectivity = True
                        continue
                    elif re.match(line.split()[0], r"!entry\.\w{3}\.unit\.hierarchy"):
                        connectivity = False
                        continue

                    if new_amino_acid:
                        atomname = line.split()[0].strip('"')
                        self.AA_ATOMS[resname].append(atomname)
                    if connectivity:
                        iatom, jatom = [int(ele) for ele in line.split()][:2]
                        self.AA_CONNECT[resname].append((iatom, jatom))

    def report_error(self):
        print("ERROR: Environment variable $AMBERHOME is not defined.")
        print("Please Checke that AmberTools and/or AmberXX should be installed and ..")
        print("       amber.sh was sourced in .zshrc")


class AmberParameter:
    """
    Class that contains Amber forcefield parameter values used to calculate potential energy.

    @ Main contents
        Atom's individual parameters: mass, polarizability, LJ parameters
        Bond energy parameters: force constant, equilibrium bond length
        Angle energy parameters: force constant, equilibrium angle
        Dihedral energy parameters: ... (See the description below.)
    """

    def __init__(
        self,
        fftype: Literal["ff99SB", "ff12SB", "ff14SB", "ff19SB"] = "ff19SB",
    ):
        assert "AMBERHOME" in os.environ, self.report_error()
        self.ATOM_PARAMS = {}
        self.BOND_PARAMS = {}
        self.ANGLE_PARAMS = {}
        self.DIHED_PARAMS = {}

        # TODO: change this for alternative options of force field. e.g ff14SB, ff19SB, etc
        self.forcefield_type = fftype
        self.forcefield_parmfile = get_forcefield_type_related_cmdfile(
            fftype=fftype, get_libfile=False, get_parmfile=True
        )

        self.parse_ff_parameters()

    def parse_ff_parameters(self) -> None:
        """
        Extract information from Amber force field parameter files.

        Parameter file specification
         - min parameter set (parm*.dat): https://ambermd.org/FileFormats.php#parm.dat
         - parameter modification file (frcmod): https://ambermd.org/FileFormats.php#frcmod

        """
        self._parse_main_parameter()
        self._parse_param_modification()

    def _parse_main_parameter(self) -> None:
        """
        Main parameter file (parm*.dat).

        reference:
        https://ambermd.org/FileFormats.php#parm.dat

        NOTE: Sectors are discriminated by a blank line

        @ Description of Sectors
        1. Atom symbols and mass
        : unique_atom_symbol   mass   atomic_polarizability(in A**3) description
        2. Bond length parameter
        : bond_atom_i-j   harmonic_force_const   equilibrium_bond_length
        3. Angle paramter (or bond angle = 3-atoms)
        : angle_atom_i-j-k   harmonic_force_const   equilibrium_angle
        4. Dihedral parameters
        : dihed_atom_i-j-k-l   IDIVF   PK   PHASE   PN
             IDIVF = factor by which the torsional barrier is divided.
                 actual torsion potential: (PK/IDIVF) * (1 * cos(PN*phi - PHASE))
             PK    = barrier height divided by a factor of 2
             PHASE = phase shift angle in the torsional function (degree unit)
             PN    = periodicity of the torsional barrier
                     NOTE: if PN < 0.0, then the torsional potential is assumed
                           to have more than one term, the value of the rest of
                           the terms are read from the next cards until a positive
                           PN is encountered.
                           The negative value of PN is used only for identifying
                           the existence of the next term and only the absolute
                           value of PN is kept.
        5. Improper dihedral parameters
        : dihed_atom_i-j-k-l   IDIVF  PK  PHASE   PN
        6. 12-6 LJ parameters
        : atom_symbol  coeff_12_power_term   coeff_6_power_term

        """
        read_atom_parm = True
        read_bond_parm = False
        read_angle_parm = False
        read_dihed_parm = False
        read_improp_parm = False
        read_lj_parm = False

        for parmfile in self.forcefield_parmfile:
            if parmfile.startswith("parm"):
                with open(parmfile) as parameter_file:
                    for line in parameter_file:
                        element = line.split()
                        if read_atom_parm:
                            if line.startswith("PARM"):
                                continue
                            if len(element) == 0:
                                read_atom_parm = False
                                read_bond_parm = True
                                continue
                            atom_type = element[0]
                            atom_mass = float(element[1])
                            try:
                                atom_polarizability = float(element[2])
                            except IndexError:
                                atom_polarizability = np.nan
                            self.ATOM_PARAMS[atom_type] = AtomParam(
                                mass=atom_mass,
                                polarizability=atom_polarizability,
                                lj_6_power_term=np.nan,
                                lj_12_power_term=np.nan,
                            )

                        if read_bond_parm:
                            if line.startswith("C   H   HO  N"):
                                continue
                            if len(element) == 0:
                                read_bond_parm = False
                                read_angle_parm = True
                                continue
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            bond_force_const = float(line[7:12].strip())
                            equil_bond_dist = float(line[14:22].strip())
                            self.BOND_PARAMS[(atom_i, atom_j)] = BondParam(
                                force_const=bond_force_const,
                                equil_bond_length=equil_bond_dist,
                            )

                        if read_angle_parm:
                            if len(element) == 0:
                                read_angle_parm = False
                                read_dihed_parm = True
                                continue
                            # in Angle parameter section, no first information line
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            atom_k = line[6:8].strip()
                            angle_force_const = float(line[8:16].strip())
                            equil_angle_degree = float(line[16:28].strip())
                            self.ANGLE_PARAMS[(atom_i, atom_j, atom_k)] = AngleParam(
                                force_const=angle_force_const,
                                equil_angle=equil_angle_degree,
                            )

                        if read_dihed_parm:
                            if len(element) == 0:
                                read_dihed_parm = False
                                read_improp_parm = True
                                continue
                            # in Dihedral parameter section, no first information line
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            atom_k = line[6:8].strip()
                            atom_l = line[9:11].strip()
                            barrier_factor = int(line[12:15].strip())
                            barrier_height = float(line[16:27].strip())
                            phase = float(line[30:40].strip())
                            periodicity = int(line[48:54].strip())
                            if (
                                atom_i,
                                atom_j,
                                atom_k,
                                atom_l,
                            ) not in self.DIHED_PARAMS:
                                if periodicity < 0:
                                    periodicity = abs(periodicity)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)] = (
                                    DihedralParam(
                                        factor_torsion_barrier_divided=barrier_factor,
                                        barrier_height=[barrier_height],
                                        phase_shift_angle=[phase],
                                        periodicity=[periodicity],
                                    )
                                )
                            else:
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "barrier_height"
                                ].append(barrier_height)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "phase_shift_angle"
                                ].append(phase)
                                if periodicity < 0:
                                    periodicity = abs(periodicity)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "periodicity"
                                ].append(periodicity)

                        if read_improp_parm:
                            if len(element) == 0:
                                read_improp_parm = False
                                continue
                            # in Improper angle section, no first information line.
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            atom_k = line[6:8].strip()
                            atom_l = line[9:11].strip()
                            barrier_height = float(line[16:27].strip())
                            phase = float(line[30:40].strip())
                            periodicity = int(line[45:52].strip())
                            if (
                                atom_i,
                                atom_j,
                                atom_k,
                                atom_l,
                            ) not in self.DIHED_PARAMS:
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)] = (
                                    DihedralParam(
                                        factor_torsion_barrier_divided=-1,
                                        barrier_height=[barrier_height],
                                        phase_shift_angle=[phase],
                                        periodicity=[periodicity],
                                    )
                                )
                            else:
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "barrier_height"
                                ].append(barrier_height)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "phase_shift_angle"
                                ].append(phase)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "periodicity"
                                ].append(periodicity)

                        if line.startswith("MOD4"):
                            read_lj_parm = True
                            continue

                        if read_lj_parm:
                            if len(element) == 0 or line.startswith("END"):
                                read_lj_parm = False
                                continue
                            atom_type = element[0]
                            lj_12_power_factor = float(element[1])
                            lj_6_power_factor = float(element[2])
                            self.ATOM_PARAMS[atom_type]["lj_12_power_term"] = (
                                lj_12_power_factor
                            )
                            self.ATOM_PARAMS[atom_type]["lj_6_power_term"] = (
                                lj_6_power_factor
                            )

    def _parse_param_modification(self):
        """
        Modified parameter file (frcmod.[forcefield_type]).

        reference:
        https://ambermd.org/FileFormats.php#parm.dat

        NOTE: Sectors are discriminated by a blank line

        @ Description of Sectors
        1. [Started with "MASS"] Atom symbols and mass
        : unique_atom_symbol   mass   atomic_polarizability(in A**3) description

        2. ["BOND"] Bond length parameter
        : bond_atom_i-j   harmonic_force_const   equilibrium_bond_length

        3. ["ANGL"] Angle paramter (or bond angle = 3-atoms)
        : angle_atom_i-j-k   harmonic_force_const   equilibrium_angle

        4. ["DIHE"] Dihedral parameters
        : dihed_atom_i-j-k-l   IDIVF   PK   PHASE   PN
             IDIVF = factor by which the torsional barrier is divided.
                 actual torsion potential: (PK/IDIVF) * (1 * cos(PN*phi - PHASE))
             PK    = barrier height divided by a factor of 2
             PHASE = phase shift angle in the torsional function (degree unit)
             PN    = periodicity of the torsional barrier
                     NOTE: if PN < 0.0, then the torsional potential is assumed
                           to have more than one term, the value of the rest of
                           the terms are read from the next cards until a positive
                           PN is encountered.
                           The negative value of PN is used only for identifying
                           the existence of the next term and only the absolute
                           value of PN is kept.

        5. ["IMPR"] Improper dihedral parameters
        : dihed_atom_i-j-k-l   IDIVF  PK  PHASE   PN

        6. ["NONB"] 12-6 LJ parameters
        : atom_symbol  coeff_12_power_term   coeff_6_power_term
        """
        read_atom_parm = False
        read_bond_parm = False
        read_angle_parm = False
        read_dihed_parm = False
        read_improp_parm = False
        read_lj_parm = False

        for parmfile in self.forcefield_parmfile:
            if parmfile.startswith("frcmod"):
                with open(parmfile) as parameter_file:
                    for line in parameter_file:
                        element = line.split()
                        if line.startswith("MASS"):
                            read_atom_parm = True
                            continue
                        if read_atom_parm:
                            try:
                                atom_type = element[0]
                                atom_mass = float(element[1])
                                atom_polarizability = float(element[2])
                            except IndexError:
                                read_atom_parm = False
                                continue
                            if atom_type not in self.ATOM_PARAMS:
                                self.ATOM_PARAMS[atom_type] = AtomParam(
                                    mass=atom_mass,
                                    polarizability=atom_polarizability,
                                    lj_12_power_term=np.nan,
                                    lj_6_power_term=np.nan,
                                )
                        if line.startswith("BOND"):
                            read_bond_parm = True
                            continue
                        if read_bond_parm:
                            if len(element) == 0:
                                read_bond_parm = False
                                continue
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            force_const = float(line[6:12].strip())
                            equil_bond_dist = float(line[16:].strip())
                            self.BOND_PARAMS[(atom_i, atom_j)] = BondParam(
                                force_const=force_const,
                                equil_bond_length=equil_bond_dist,
                            )
                        if line.startswith("ANGL"):
                            read_angle_parm = True
                            continue
                        if read_angle_parm:
                            if len(element) == 0:
                                read_angle_parm = False
                                continue
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            atom_k = line[6:8].strip()
                            force_const = float(line[11:16].strip())
                            equil_angle = float(line[22:].strip())
                            self.ANGLE_PARAMS[(atom_i, atom_j, atom_k)] = AngleParam(
                                force_const=force_const, equil_angle=equil_angle
                            )
                        if line.startswith("DIHE"):
                            read_dihed_parm = True
                            continue
                        if read_dihed_parm:
                            if len(element) == 0:
                                read_dihed_parm = False
                                continue
                            # in Dihedral parameter section, no first information line
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            atom_k = line[6:8].strip()
                            atom_l = line[9:11].strip()
                            barrier_factor = int(line[12:15].strip())
                            barrier_height = float(line[16:24].strip())
                            phase = float(line[29:36].strip())
                            periodicity = int(line[47:51].strip())
                            if (
                                atom_i,
                                atom_j,
                                atom_k,
                                atom_l,
                            ) not in self.DIHED_PARAMS:
                                if periodicity < 0:
                                    periodicity = abs(periodicity)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)] = (
                                    DihedralParam(
                                        factor_torsion_barrier_divided=barrier_factor,
                                        barrier_height=[barrier_height],
                                        phase_shift_angle=[phase],
                                        periodicity=[periodicity],
                                    )
                                )
                            else:
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "barrier_height"
                                ].append(barrier_height)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "phase_shift_angle"
                                ].append(phase)
                                if periodicity < 0:
                                    periodicity = abs(periodicity)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "periodicity"
                                ].append(periodicity)
                        if line.startswith("IMPR"):
                            read_improp_parm = True
                            continue
                        if read_improp_parm:
                            if len(element) == 0:
                                read_improp_parm = False
                                continue
                            # in Improper angle section, no first information line.
                            atom_i = line[:2].strip()
                            atom_j = line[3:5].strip()
                            atom_k = line[6:8].strip()
                            atom_l = line[9:11].strip()
                            barrier_height = float(line[16:27].strip())
                            phase = float(line[30:40].strip())
                            periodicity = int(line[45:].strip())
                            if (
                                atom_i,
                                atom_j,
                                atom_k,
                                atom_l,
                            ) not in self.DIHED_PARAMS:
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)] = (
                                    DihedralParam(
                                        factor_torsion_barrier_divided=-1,
                                        barrier_height=[barrier_height],
                                        phase_shift_angle=[phase],
                                        periodicity=[periodicity],
                                    )
                                )
                            else:
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "barrier_height"
                                ].append(barrier_height)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "phase_shift_angle"
                                ].append(phase)
                                self.DIHED_PARAMS[(atom_i, atom_j, atom_k, atom_l)][
                                    "periodicity"
                                ].append(periodicity)
                        if line.startswith("NONB"):
                            read_lj_parm = True
                            continue
                        if read_lj_parm:
                            if len(element) == 0:
                                read_lj_parm = False
                                continue
                            atom_type = element[0]
                            lj_12_power_factor = float(element[1])
                            lj_6_power_factor = float(element[2])
                            self.ATOM_PARAMS[atom_type]["lj_12_power_term"] = (
                                lj_12_power_factor
                            )
                            self.ATOM_PARAMS[atom_type]["lj_6_power_term"] = (
                                lj_6_power_factor
                            )

    def report_error(self):
        print("ERROR: Environment variable $AMBERHOME is not defined.")
        print("Please Checke that AmberTools and/or AmberXX should be installed and ..")
        print("       amber.sh was sourced in .zshrc")


if __name__ == "__main__":
    top = AmberTopology()
    parm = AmberParameter()
