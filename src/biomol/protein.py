from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from Bio.PDB import Residue
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from biomol import coordinate, rotamer
from biomol.topology import AmberParameter, AmberTopology


class ChainSelect(Select):
    """
    For selecting a specific chain id's part of Bio.PDB.Model.
    """

    def __init__(self, chainid):
        self.chainID = chainid

    def accept_chain(self, chain):
        return chain.id == self.chainID


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

        Protein.check_pdb_file(self.pdbfile_path)
        pdb_parser = PDBParser(QUIET=True)

        self.pdb_str = pdb_parser.get_structure("input_protein", self.pdbfile_path)
        assert self.pdb_str is not None
        # remove water molecules
        # remove hetero residue only chains
        if remove_wat:
            for model in self.pdb_str:
                for chain in list(model):
                    is_het_only_chain = True
                    for residue in list(chain):
                        if residue.get_id()[0] == "W" or residue.get_resname() == "HOH":
                            chain.detach_child(residue.id)
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
                #       self.residues : {'chain_id': {resid_i: residue_obj}}
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
        assert file_path.endswith(".pdb"), "File must have .pdb extension"
        assert Path(file_path).is_file(), f"{file_path} is not a valid file"

    @staticmethod
    def util_upper_chainids(chids: str | list):
        if type(chids) is str:
            return chids.upper()
        else:
            # type(chids) is list
            return [c.upper() for c in chids]

    def get_num_chain(self) -> int:
        return len(self.residues)

    def get_chain_ids(self) -> list:
        return list(self.residues.keys())

    def get_residues(self) -> dict:
        return self.residues

    def get_residue_ids(self, chainid: str | None = None) -> list | dict:
        if chainid is None:
            resids = {}
            for chid in self.residues:
                resids[chid] = list(self.residues[chid].keys())
            return resids
        else:
            return list(self.residues[chainid].keys())

    def get_crds_list(self, chainid: str | list | None = None) -> list:
        crdlist = []
        if chainid is not None:
            chainid = Protein.util_upper_chainids(chainid)

        # res: Biopython.PDB.Residue
        for chid, reslist in self.residues.items():
            if chainid is not None and chid.upper() not in chainid:
                continue
            for res in reslist:
                for atm in res:
                    crdlist.append(atm.get_coord())
        return crdlist

    def get_residue_of_specific_resid(self, resid: int, chainid: str | None = None):
        if chainid is not None:
            return self.residues[chainid][resid]
        else:
            residue_list = []
            for ch in self.residues:
                if resid in self.residues[ch]:
                    residue_list.append(self.residues[ch][resid])
            return residue_list

    def get_atom_crds_and_their_resids(self, chainid: str | list | None = None) -> list:
        crdlist = []
        if chainid is not None:
            chainid = Protein.util_upper_chainids(chainid)

        for chid, reslist in self.residues.items():
            if chainid is not None and chid.upper() not in chainid:
                continue
            for res in reslist:
                for atm in res:
                    tmp = []
                    tmp.extend(atm.get_coord())
                    tmp.append(res.get_id()[1])
                    crdlist.append(tmp)
        return crdlist

    def extract_seqres_records(self) -> None:
        with open(self.pdbfile_path) as pdbfile:
            for line in pdbfile:
                if line.startswith("SEQRES"):
                    chainid = line[11]
                    if chainid not in self.seqres:
                        self.seqres[chainid] = []
                    self.seqres[chainid].append(line)

    def extract_all_chains(self) -> None:
        # name, ext = osp.splitext(self.pdbfile_path)
        name = Path(self.pdbfile_path).stem

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
                chain_sasa[chain.id][residue.id] = residue.sasa
        residue_sasa = {}  # SASA of residues when it exists as a single amino acid with the same conformation
        for chain in self.pdb_model:
            residue_sasa[chain.id] = {}
            for residue in chain:
                sr.compute(residue, level="R")
                residue_sasa[chain.id][residue.id] = residue.sasa
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
        assert isinstance(
            resid2resobj_dict, dict
        ), f"{chainid} does not exist in this protein model."
        key1 = next(iter(resid2resobj_dict))
        first_residue = resid2resobj_dict[key1]
        assert isinstance(
            first_residue, Residue.Residue
        ), "Residue.Residue objects should be in the resid2resobj_dict"

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
        self, *, resid: int, chid: str, wt_res: str, mut_res: str
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
            self.rotamer_library, mut_res, phi, psi, sort_by="Probabil", rank=0
        )
        # Step 4.
        # Step 4-1.
        copied_backbone_crd = coordinate.copy_backbone_coordinate(
            residue_obj=self.residues[chid][resid]
        )
        # Step 4-2-1
        # Step 4-2-2
        mutres_int_coord = coordinate.construct_residue_internal_coord_of_sidechain(
            resname=mut_res,
            forcefield_topology=self.topology,
            forcefield_parameter=self.parameter,
            chi_angles=sidechain_rotamers,
        )
        # Step 4-2-3
        mutres_cart_coord = coordinate.convert_internal_to_cartesian_coordinate(
            resname=mut_res,
            zmatrix=mutres_int_coord,
            backbone_crd=copied_backbone_crd,
            forcefield_topology=self.topology,
        )
        # Construct Atoms

        return mutres_cart_coord


if __name__ == "__main__":
    pdbfile = "1acbE.pdb"
    protein = Protein(pdbfile)

    resid = 54
    chainid = "A"
    residue = protein.get_residue_of_specific_resid(resid=resid, chainid=chainid)
    print(f"Number of residue: {protein.get_residues()}")
    # print(f"Phi of resid-{resid} of chain-{chainid}: {residue.get_angle('phi')}")
