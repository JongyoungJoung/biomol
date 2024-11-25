import os
import sys
from os import path as osp
from pathlib import Path

import numpy as np
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from rotamer import init_rotamer
from tqdm import tqdm


class ChainSelect(Select):
    def __init__(self, chainID):
        self.chainID = chainID

    def accept_chain(self, chain):
        return chain.id == self.chainID


class Protein:
    def __init__(
        self,
        pdbfile_path: str,
        chid: str | None = None,
        remove_wat: bool | None = True,
        # remove_het_only_chain: bool | None = True,
    ):
        self.pdbfile_path = pdbfile_path
        self.residues = {}
        self.residue_ids = {}
        self.exposed_sasa_ratio = {}
        self.detected_surface_residues = {}
        self.seqres = {}

        Protein.check_pdb_file(self.pdbfile_path)
        # pdb_parser = PDBParser(PERMISSIVE=1)
        pdb_parser = PDBParser(QUIET=True)

        self.pdb_str = pdb_parser.get_structure("input_protein", self.pdbfile_path)
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
                    self.residues[this_chid] = []
                for residue in chain:
                    self.residues[this_chid].append(residue)

        for chid in self.residues:
            if chid not in self.residue_ids:
                self.residue_ids[chid] = []
            for residue in self.residues[chid]:
                self.residue_ids[chid].append(residue.get_id()[1])

        # parser SEQRES parts
        self.extract_seqres_records()

    @staticmethod
    def check_pdb_file(file_path) -> None:
        assert isinstance(file_path, str), "File path must be a string"
        assert file_path.endswith(".pdb"), "File must have .pdb extension"
        assert Path(file_path).is_file(), f"{file_path} is not a valid file"

    @staticmethod
    def util_upper_chainIds(chids: str | list):
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

    def get_residue_ids(self) -> dict:
        return self.residue_ids

    def get_crds_list(self, given_chid: str | list | None = None) -> list:
        crdlist = []
        if given_chid is not None:
            given_chid = Protein.util_upper_chainIds(given_chid)

        # res: Biopython.PDB.Residue
        for chid, reslist in self.residues.items():
            if given_chid is not None and chid.upper() not in given_chid:
                continue
            for res in reslist:
                for atm in res:
                    crdlist.append(atm.get_coord())
        return crdlist

    def get_crds_list_with_resid(self, given_chid: str | list | None = None) -> list:
        crdlist = []
        if given_chid is not None:
            given_chid = Protein.util_upper_chainIds(given_chid)

        for chid, reslist in self.residues.items():
            if given_chid is not None and chid.upper() not in given_chid:
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

    # Set mutation using rotamer library
    #
    def mutate(self, resid: int, chid: str, wt_res: str, mut_res: str) -> None:
        pass


if __name__ == "__main__":
    pdbfile = "1acbE.pdb"
    protein = Protein(pdbfile)

    print(f"Number of residue: {protein.get_residues()}")
