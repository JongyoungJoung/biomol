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
from pathlib import Path
from typing import Literal


class AmberTopology:
    def __init__(
        self,
        fftype: Literal["ff99SB", "ff12SB", "ff14SB", "ff19SB"] = "ff19SB",
    ):
        assert "AMBERHOME" in os.environ, self.report_error()
        self.AA_ATOMS = {}
        self.AA_CONNECT = {}

        # TODO: change this for alternative options of force field. e.g ff14SB, ff19SB, etc
        self.forcefield_type = fftype
        libfile = AmberTopology.match_forcefield_libfile(self.forcefield_type)
        self.forcefield_libfile = f"{os.environ('AMBERHOME')}/dat/leap/lib/{libfile}"
        assert Path(
            self.forcefield_libfile
        ).exists, "Forcefield library file doest not exist."

        self.parse_ff_library()

    @staticmethod
    def match_forcefield_libfile(ff_type: str) -> str | os.PathLike:
        libfiles = {
            "ff99SB": "oldff/all_amino91.lib",
            "ff12SB": "amino12.lib",
            "ff14SB": "amino14ipq.lib",
            "ff19SB": "amino19.lib",
        }

        return libfiles[ff_type]

    def parse_ff_library(self):
        """
        Parse Amber forcefield library file which defines topology of amino acids.
        """
        new_amino_acid = False
        connectivity = False
        resname = ""
        atomname = ""
        with open(self.forcefield_libfile) as libfile:
            for line in libfile:
                if re.match(line.split()[0], r"!entry\.\w{3}\.unit\.atomspertinfo"):
                    new_amino_acid = True
                    resname = line.split()[0].split(".")[1]
                    self.AA_ATOMS[resname] = []
                    continue
                elif re.match(line.split()[0], r"!entry\.\w{3}\.unit\.boundbox"):
                    new_amino_acid = False
                    continue
                elif re.match(line.split()[0], r"!entry\.\w{3}\.unit\.connectivity"):
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

    # @classmethod
    # def setup_internal_coord(cls, ...

    def report_error(self):
        print("ERROR: Environment variable $AMBERHOME is not defined.")
        print("Please Checke that AmberTools and/or AmberXX should be installed and ..")
        print("       amber.sh was sourced in .zshrc")


if __name__ == "__main__":
    top = AmberTopology()
