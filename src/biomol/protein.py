from functools import reduce
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio.PDB import Atom, Residue
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from biomol import biomol_residue as bioresidue
from biomol import coordinate, rotamer
from biomol.libs import protein_util
from biomol.topology import AmberParameter, AmberTopology


class ChainSelect(Select):
    """
    For selecting a specific chain id's part of Bio.PDB.Model.
    """

    def __init__(self, chainid):
        self.chainID = chainid

    def accept_chain(self, chain):
        return chain.id == self.chainID


class ChainTerSelect(Select):
    """
    For selecting all pdb lines including TER lines when extracted into a pdb file.
    """

    def __init__(self):
        self.prev_chain_id = None

    def accept_residue(self, residue):  # type: ignore
        # Accept  all residues
        return True


class Protein:
    """
    Protein Class.
    """

    def __init__(
        self,
        pdbfile_path: str,
        chid: str | None = None,
        *,
        remove_wat: bool | None = True,
        load_ff: bool | None = False,
        ff_type: Literal["ff99SB", "ff12SB", "ff14SB", "ff19SB"] = "ff19SB",
        # remove_het_only_chain: bool | None = True,
    ):
        """
        Instance variables.

        - self.residues: {chainid: {resid: residueObj, ..}  Key parsed information of a protein

        """
        # basic information
        self.pdbfile_path = pdbfile_path
        self.residues = {}
        self.num_atoms = 0
        self.num_residues = 0
        self.ssbond_pair = {}
        # processed information
        self.exposed_sasa_ratio = {}
        self.detected_surface_residues = {}
        self.seqres = {}
        self.rotamer_library = pd.DataFrame()
        # force field related.
        self.forcefield_loaded = False
        self.forcefield_type = ff_type
        self.topology: AmberTopology
        self.parameter: AmberParameter
        # modification
        self.mutated_residue: list[Residue.Residue] = []

        Protein.check_pdb_file(self.pdbfile_path)
        pdb_parser = PDBParser(QUIET=True)

        self.pdb_str = pdb_parser.get_structure("input_protein", self.pdbfile_path)
        assert self.pdb_str is not None, (
            "Failed to generate Biopython.PDB object when reading input pdb file"
        )
        # remove water molecules
        # remove hetero residue only chains
        for model in self.pdb_str:
            for chain in list(model):
                is_het_only_chain = True
                for residue in list(chain):
                    # if residue.get_id()[0] == "W" or residue.get_resname() == "HOH":
                    if remove_wat and (
                        residue.get_id()[0] == "W"
                        or bioresidue.is_solvent(residue.get_resname())
                    ):
                        chain.detach_child(residue.id)
                        continue
                    # count up number of atoms
                    for _ in residue:
                        self.num_atoms += 1
                    if residue.get_id()[0] == " ":
                        # if any nomal amino acid appear
                        is_het_only_chain = False
                if is_het_only_chain:
                    model.detach_child(chain.id)

        self.pdb_model = self.pdb_str[0]

        if chid is not None:
            self.residues[chid] = self.pdb_model[chid]
        else:
            for chain in self.pdb_model:
                this_chid = chain.get_id()
                if this_chid not in self.residues:
                    self.residues[this_chid] = {}
                # NOTE: For convenient hashing residues by residue id
                #       self.residues : {'chain_id': {resid_id: residue_obj}}
                for residue in chain:
                    self.residues[this_chid][residue.get_id()[1]] = residue

        # parser SEQRES parts
        self.extract_seqres_records()

        # Loading force field
        if load_ff:
            self.init_forcefield()

    @staticmethod
    def check_pdb_file(file_path) -> None:
        assert isinstance(file_path, str), "File path must be a string"
        assert Path(file_path).suffix == ".pdb", "File must have .pdb extension"
        assert Path(file_path).is_file(), f"{file_path} is not a valid file"

    @staticmethod
    def util_upper_chainids(chids: str | list):
        if type(chids) is str:
            return chids.upper()
        else:
            # type(chids) is list
            return [c.upper() for c in chids]

    @staticmethod
    def get_atomic_crds_of_residue(residue: Residue.Residue) -> list:
        res_crds = []
        for atom in residue:
            res_crds.append(atom.get_coord())
        return res_crds

    @staticmethod
    def get_residue_formal_charge(
        *,
        residue: Residue.Residue,
        is_N_terminal: bool = False,
        is_C_terminal: bool = False,
    ) -> int | float:
        """
        Return a residue's formal net charge.
        """
        metal_ions = ["ZN", "MG", "FE", "CA", "MN"]
        res_charge = 0.0
        resname = residue.get_resname()
        if resname in ["ARG", "LYS", "HIP"]:
            res_charge = 1.0
        elif resname in ["ASP", "GLU", "CYM"]:
            res_charge = -1.0
        elif resname in ["SEP", "TPO"]:
            res_charge = -2.0
        elif resname in ["GDP", "ADP"]:
            res_charge = -3.0
        elif resname in ["GTP", "GNP", "GCP", "GSP", "ATP", "ACP"]:
            res_charge = -4.0
        elif any(metal in resname for metal in metal_ions):
            res_charge = -2.0
        else:
            res_charge = 0.0

        # NOTE: if a residue is a single amino acid, it can be both N-terminal and C-terminal.
        if is_N_terminal:
            res_charge += 1.0
        if is_C_terminal:
            res_charge -= 1.0

        return res_charge

    def get_num_atoms(self) -> int:
        return self.num_atoms

    def get_num_chain(self) -> int:
        return len(self.residues)

    def get_chain_ids(self) -> list:
        return list(self.residues.keys())

    def get_residues(self) -> dict:
        """
        Return residues dictionary.

        Return value:
            {chain_id: {residues_id(=original id) : Bio.PDB.Residue.Residue object}, ...}
        """
        return self.residues

    def get_num_residues(self) -> int:
        self.num_residues = 0
        for _, chain_residues in self.residues.items():
            self.num_residues += len(chain_residues)

        return self.num_residues

    def get_residues_as_one_dict(self, chid: str | None = None) -> dict:
        if chid is None:
            # NOTE: if  chid is not given. choose all chains' residues.
            all_ = [residues for _, residues in self.residues.items()]
            all_residues = {}
            if self.get_num_chain() > 1:
                # 1. first check whether residues' names of multiple chains are redundant
                common_resname = set(all_[0]).intersection(
                    *(d.keys() for d in all_[1:])
                )
                if len(common_resname) >= 1:
                    # if at least one common residue ids are found, replace all the residue name
                    for ch, residues in self.residues.items():
                        for res_id, res_obj in residues.items():
                            all_residues[f"{ch}_{res_id}"] = res_obj
                else:
                    # if all residue ids are unique, just merge all chain dictionary of residues
                    all_residues = reduce(lambda a, b: a | b, all_)

                return all_residues
            else:
                # if this protein has a single chain.
                return all_[0]
        else:
            # NOTE: If specific chain id is given.
            return self.residues[chid]

    def get_residue_ids(self, chainid: str | None = None) -> list | dict:
        if chainid is None:
            resids = {}
            for chid in self.residues:
                resids[chid] = list(self.residues[chid].keys())
            return resids
        else:
            return list(self.residues[chainid].keys())

    def get_crds_list(
        self, *, chainid: str | list | None = None, as_list: bool = False
    ) -> list | npt.NDArray:
        """
        Get all the coordinates of this chain or protein.
        """
        crdlist: list | npt.NDArray = []
        if chainid is not None:
            chainid = Protein.util_upper_chainids(chainid)

        # res: Biopython.PDB.Residue
        for chid, resdict in self.residues.items():
            # NOTE: If chainid is available, Only residue crds from the selected chain.
            if chainid is not None and chid.upper() not in chainid:
                continue
            for _, resobj in resdict.items():
                for atm in resobj:
                    crdlist.append(atm.get_coord())
        if not as_list:
            crdlist = np.array(crdlist)

        return crdlist

    def get_residue_of_specific_resid(self, resid: int, chainid: str | None = None):
        # FIXME:
        # Way to indicate a specific residue should be updated.
        if chainid is not None:
            return self.residues[chainid][resid]
        else:
            residue_list = []
            for ch in self.residues:
                if resid in self.residues[ch]:
                    residue_list.append(self.residues[ch][resid])
            return residue_list

    def get_atom_crds_and_their_resids(self, chainid: str | list | None = None) -> list:
        """
        Get atoms' XYZ coordinates and their residue ids.

        Return:
            [[(x, y, z), resid], ....]
        """
        crdlist = []
        if chainid is not None:
            chainid = Protein.util_upper_chainids(chainid)

        for chid, resdict in self.residues.items():
            if chainid is not None and chid.upper() not in chainid:
                continue
            for resid, res in resdict.items():
                for atm in res:
                    tmp = []
                    tmp.extend(atm.get_coord())
                    # resid == res.get_id()[1]
                    tmp.append(res.get_id()[1])
                    # NOTE: [(X, Y, Z), Resid]
                    crdlist.append(tmp)

        return crdlist

    def search_ss_bond(self, *, ssbond_dist: float = 2.05) -> bool:
        """
        Search CYS residues and check disulfide bond between SG atoms.
        """
        found: bool = False
        # NOTE: search CYS residues
        cys_sg_dict = {}
        for chid, resdict in self.residues.items():
            for resid, res in resdict.items():
                if res.get_resname() in ["CYS", "CYX"]:
                    for atm in res:
                        if atm.get_name() == "SG":
                            cys_sg_dict[(chid, resid)] = atm.get_coord()

        # NOTE: check SS bond
        cys_sg_ids = list(cys_sg_dict.keys())
        for i in range(len(cys_sg_ids) - 1):
            sg_1_id = cys_sg_ids[i]
            sg_1_crd = cys_sg_dict[sg_1_id]

            for j in range(i + 1, len(cys_sg_ids)):
                sg_2_id = cys_sg_ids[j]
                sg_2_crd = cys_sg_dict[sg_2_id]

                if np.linalg.norm(sg_1_crd - sg_2_crd) <= ssbond_dist:
                    self.ssbond_pair[sg_1_id] = sg_2_id
                    self.ssbond_pair[sg_2_id] = sg_1_id
        if self.ssbond_pair:
            found = True

        return found

    def get_ssbond_pair(self) -> dict:
        """
        Get SS-bond pairs.

        Return:
            {(chid_1, resid_1):(chid_2, resid_2),
             (chid_2, resid_2):(chid_1, resid_1)
            ...}
        """
        return self.ssbond_pair

    def extract_seqres_records(self) -> None:
        """
        Extract SEQRES records from the input pdb file.
        """
        with open(self.pdbfile_path) as pdbfile:
            for line in pdbfile:
                if line.startswith("SEQRES"):
                    chainid = line[11]
                    if chainid not in self.seqres:
                        self.seqres[chainid] = []
                    self.seqres[chainid].append(line)

    def extract_as_one_pdbfile(self, *, outpdb_name: str | None = None) -> None:
        """
        Extract pdb object into one pdb file.
        """
        if outpdb_name is None:
            name = Path(self.pdbfile_path).stem
        else:
            name = Path(outpdb_name).stem

        io = PDBIO()
        io.set_structure(self.pdb_str)
        io.save(f"{name}.pdb", select=ChainTerSelect())

    def extract_all_chains(self, *, outpdb_name: str | None = None) -> None:
        """
        Extract all chains into individual pdb files.
        """
        # name, ext = osp.splitext(self.pdbfile_path)
        if outpdb_name is None:
            name = Path(self.pdbfile_path).stem
        else:
            name = Path(outpdb_name).stem

        io = PDBIO()
        io.set_structure(self.pdb_str)
        for chain_obj in self.pdb_model:
            cid = chain_obj.id
            chain_outfile = f"{name}{cid}.pdb"
            if len(self.seqres[cid]) != 0:
                with open(chain_outfile, "w") as out_file:
                    for record in self.seqres[cid]:
                        out_file.write(record)
            with open(chain_outfile, "a") as out_file:
                io.save(out_file, select=ChainSelect(cid))

    def extract_specific_chains(self, chains: str | list) -> None:
        """
        Extract specific chains into pdb files.
        """
        # name, ext = osp.splitext(self.pdbfile_path)
        name = Path(self.pdbfile_path).stem
        io = PDBIO()
        io.set_structure(self.pdb_str)
        target_cids = "".join(chains)
        chain_outfile = f"{name}{target_cids}.pdb"
        if Path(chain_outfile).exists():
            Path(chain_outfile).unlink()

        for cid in target_cids:
            chain_obj = self.pdb_model[cid]
            if len(self.seqres[cid]) != 0:
                with open(chain_outfile, "a") as out_file:
                    for record in self.seqres[cid]:
                        out_file.write(record)
            with open(chain_outfile, "a") as out_file:
                io.save(out_file, select=ChainSelect(cid))

    # surface detection
    def detect_surface_residues(self, expose_cutoff: float = 0.01) -> None:
        # using "Rolling ball" algorithm by Sharke & Rupley
        # implemented in biopython
        sr = ShrakeRupley()
        chain_sasa = {}  # SASA of residues in 3D structure
        for chain in self.pdb_model:
            sr.compute(chain, level="R")
            chain_sasa[chain.id] = {}
            for residue in chain:
                chain_sasa[chain.id][residue.id] = residue.sasa  # type: ignore
        residue_sasa = {}  # SASA of residues when it exists as a single amino acid with the same conformation
        for chain in self.pdb_model:
            residue_sasa[chain.id] = {}
            for residue in chain:
                sr.compute(residue, level="R")
                residue_sasa[chain.id][residue.id] = residue.sasa  # type: ignore
        # get SASA ratio (sasa of "in_fold" / sasa of "alone")
        for chain in self.pdb_model:
            self.exposed_sasa_ratio[chain.id] = {}
            self.detected_surface_residues[chain.id] = []
            for residue in chain:
                # percentage of exposed SASA of a residue in a 3D protein fold compared to its single amino acid form
                sasa_ratio = (
                    chain_sasa[chain.id][residue.id]
                    / residue_sasa[chain.id][residue.id]
                )
                self.exposed_sasa_ratio[chain.id][residue.id] = sasa_ratio
                if sasa_ratio >= expose_cutoff:
                    self.detected_surface_residues[chain.id].append(residue.id[1])

    def check_residue_in_surface(self, rid: int, reschid: str | None = None) -> bool:
        is_in_surface = False
        if len(self.exposed_sasa_ratio) == 0:
            self.detect_surface_residues()
        if reschid is not None:
            chains = [reschid]
        else:
            chains = self.get_chain_ids()
        for cid in chains:
            if rid in self.detected_surface_residues[cid]:
                is_in_surface = True
                break

        return is_in_surface

    def init_rotamer_library(self):
        """
        Read rotamer library file and load it in pd.Dataframe.
        """
        self.rotamer_library = rotamer.load_rotamer_library()

    def init_forcefield(self):
        """
        Load force field related informations.

        - Reading topology (amino aicds') files
        - Reading force field parameters (TODO)
        """
        if not self.forcefield_loaded:
            self.topology = AmberTopology()
            self.parameter = AmberParameter()
            self.forcefield_loaded = True

    def calc_dihedral_angle_of_residue_with_given_type(
        self, dihedral_type: str, resid: int, chainid: str | None = None
    ) -> float:
        """
        Calculate dihedral angles of given type for a residue of resid.
        """
        dihedral = 0.0

        resid2resobj_dict = None
        if chainid is not None:
            resid2resobj_dict = self.residues[chainid]
        else:
            for ch in self.residues:
                if resid in self.residues[ch]:
                    """If chainid is None, select residue of resid found first in chain list"""
                    resid2resobj_dict = self.residues[ch]
                    break
        assert isinstance(resid2resobj_dict, dict), (
            f"{chainid} does not exist in this protein model."
        )
        key1 = next(iter(resid2resobj_dict))
        first_residue = resid2resobj_dict[key1]
        assert isinstance(first_residue, Residue.Residue), (
            "Residue.Residue objects should be in the resid2resobj_dict"
        )

        resname = resid2resobj_dict[resid].get_resname()
        prev_res = None
        next_res = None
        if resid - 1 not in resid2resobj_dict:
            n_term = True
        else:
            n_term = False
            if dihedral_type.upper() == "PHI":
                prev_res = resid2resobj_dict[resid - 1]
        if resid + 1 not in resid2resobj_dict:
            c_term = True
        else:
            c_term = False
            if dihedral_type.upper() in ["PSI", "OMEGA"]:
                next_res = resid2resobj_dict[resid + 1]

        dihedral = rotamer.calc_dihedral_angle(
            resid2resobj_dict[resid],
            dihedral_type,
            resname,
            prev_residue_obj=prev_res,
            next_residue_obj=next_res,
            n_terminal=n_term,
            c_terminal=c_term,
        )
        return dihedral

    def mutate(
        self, *, resid: int, chid: str, mut_res_name: str
    ) -> dict[str, np.ndarray]:
        """
        Set mutation on a specific residue.

        1. Check rotamer library loaded already.
        2. Calculate backbone dihedral angles "phi" and "psi".
        3. Find the most probable chi1 ~ chi4 angles.
        4. Construct atoms' positions of the new residue (mutant).
          4-1. Copy the wild type residue's backbone information (atom types, coordinate)
                    to mutant's backbone.
          4-2. Construct atomic positions
            4-2-1. Get a predefined internal coordinate (IC) template of
                        atomic composition of a residue.
            4-2-2. Change IC of a side chain part according to IC selected in step-3.
            4-2-3. Convert IC to Cartesian coordinates (CC) (ic2cc method required)
        """
        # FIX:
        #  @ 2025.07.01. Mutation function can't work properly
        #                in generating internal coordinates from rotamer library

        mut_res_name = protein_util.convert_residue_name_to_3_letter(
            resname=mut_res_name
        )
        # Step 1.
        # if self.rotamer_library is empty (== not initialized yet)
        # RFE:
        # Is it good way to have rotamer library as instance variable??
        if self.rotamer_library.size == 0:
            self.init_rotamer_library()
        if not self.forcefield_loaded:
            self.init_forcefield()

        # Step 2.
        phi = self.calc_dihedral_angle_of_residue_with_given_type(
            dihedral_type="phi", resid=resid, chainid=chid
        )
        psi = self.calc_dihedral_angle_of_residue_with_given_type(
            dihedral_type="psi", resid=resid, chainid=chid
        )
        # Step 3.
        sidechain_rotamers = rotamer.get_sidechain_rotamers(
            self.rotamer_library, mut_res_name, phi, psi, sort_by="Probabil", rank=0
        )
        # Step 4.
        # Step 4-1.
        copied_backbone_crd = coordinate.copy_backbone_coordinate(
            residue_obj=self.residues[chid][resid]
        )
        # Step 4-2-1
        # Step 4-2-2
        mutres_int_coord = coordinate.construct_residue_internal_coord_of_sidechain(
            resname=mut_res_name,
            forcefield_topology=self.topology,
            forcefield_parameter=self.parameter,
            chi_angles=sidechain_rotamers,
        )
        # Step 4-2-3
        mutres_cart_coord = coordinate.convert_internal_to_cartesian_coordinate(
            resname=mut_res_name,
            zmatrix=mutres_int_coord,
            backbone_crd=copied_backbone_crd,
            forcefield_topology=self.topology,
        )
        # Construct Atoms
        mutresidue = self.build_new_residue(
            resid=resid, resname=mut_res_name, atom_name_crd_dict=mutres_cart_coord
        )

        return mutres_cart_coord

    def build_new_residue(
        self,
        *,
        resid: int,
        resname: str,
        atom_name_crd_dict: dict[str, np.ndarray],
        default_atom_bfactor: float = 30.0,
        default_atom_occupancy: float = 1.0,
        last_atom_serial_num: int = 0,
    ) -> Residue.Residue:
        """
        Build new residue object of Biopython with atom_name and crd pairs.

        Args:
            resid: residue index
            resname: residue name (3 letters)
            atom_name_crd_dict: atom_name and crd pair
            default_atom_bfactor: atomic bfactor (default)
            default_atom_occupancy: atomic occupancy (default)
            last_atom_serial_num: last atom's serial number of previous residue,
                                  needed for continous atomic serial number

        Returns:
            Biopython Residue class object
        """
        res_id = (" ", resid, " ")
        new_residue = Residue.Residue(res_id, resname, " ")

        for atom_name, crd in atom_name_crd_dict.items():
            last_atom_serial_num += 1
            new_atom = Atom.Atom(
                name=atom_name,
                coord=crd,
                bfactor=default_atom_bfactor,
                occupancy=default_atom_occupancy,
                element=atom_name[0],
                altloc=" ",
                fullname=atom_name,
                serial_number=last_atom_serial_num,
            )
            new_residue.add(new_atom)

        return new_residue


if __name__ == "__main__":
    pdbfile = "1acbE.pdb"
    protein = Protein(pdbfile)

    resid = 54
    chainid = "A"
    residue = protein.get_residue_of_specific_resid(resid=resid, chainid=chainid)
    print(f"Number of residue: {protein.get_residues()}")
    # print(f"Phi of resid-{resid} of chain-{chainid}: {residue.get_angle('phi')}")
