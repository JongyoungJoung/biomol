# import shutil
from pathlib import Path
from typing import Literal

import numpy as np

from biomol import biomol_residue, ligand, protein


class ProteinLigandComplex:
    """
    Protein-ligand complex class.

    -------
    Inputs:
    pdb file of protein-ligand complex

    -------
    Tasks:
    - split input pdb into protein pdb and ligand pdb
    """

    def __init__(
        self,
        *,
        inpdb: str,
        pdbname: str | None = None,
        preserve_intermediate_files: bool = False,
        desolvate: bool = False,
        verbose: bool = False,
    ):
        self.inpdb = inpdb
        self.pdbname: str
        if pdbname is None:
            self.pdbname = Path(self.inpdb).stem
        else:
            self.pdbname = Path(pdbname).name
        self.receptor_pdbname: str = ""
        self.ligand_pdbname: str = ""
        self.n_ligand = 0
        self.n_water = 0
        self.first_residue_resid: int | None = None
        self.receptor_obj: protein.Protein
        self.ligand_obj: ligand.Ligand

        # NOTE: Separate proteinn part and ligand part, and generate splitted pdb files
        self.split_complex()
        self.load_protein_ligand(
            preserve_intermediates=preserve_intermediate_files,
            verbose=verbose,
            desolvate=desolvate,
        )
        # by default, set all the protein residues as selected residue
        self.is_selected_residues = [True] * self.receptor_obj.get_num_residues()

    def split_complex(self) -> None:
        """
        Split complex pdb into protein pdb and ligand pdb.

        Outputs:
            - protein pdb file
            - ligand pdb file
        """
        protein_part = []
        ligand_part = []
        protein_atom_id = []
        ligand_atom_id = []
        prev_resname = ""
        prev_resid = -1
        prev_atmid = -1
        prev_org_resid = -1
        with open(self.inpdb) as inpdb:
            for line in inpdb:
                if line.startswith(("ATOM", "HETATM")):
                    resname = line[17:20].strip()
                    atid = int(line[6:11].strip())
                    resid = int(line[22:26].strip())
                    if self.first_residue_resid is None:
                        self.first_residue_resid = resid
                        prev_resid = resid

                    if biomol_residue.is_amino_acid(resname):
                        prev_atmid = atid
                        if prev_resid != resid:
                            prev_resid = resid
                    elif biomol_residue.is_water(resname):
                        prev_atmid = prev_atmid + 1
                        if prev_org_resid != resid:
                            self.n_water += 1
                            prev_org_resid = resid
                            prev_resid = prev_resid + 1
                            # if water... renumber resid of waters
                        line = f"{line[:6]}{prev_atmid:>5d}{line[11:22]}{prev_resid + 1:>4d}{line[26:]}"

                    # distinguish protein lines and ligand lines
                    if (
                        biomol_residue.is_amino_acid(resname)
                        or biomol_residue.is_water(resname)
                        # or biomol_residue.is_metal_ion(resname)
                    ):
                        protein_part.append(line)
                        protein_atom_id.append(atid)
                    elif not biomol_residue.is_neutral_ion(resname):
                        ligand_part.append(line)
                        ligand_atom_id.append(atid)
                        if resname != prev_resname:
                            self.n_ligand += 1
                    prev_resname = resname
                elif line.startswith("TER"):
                    if biomol_residue.is_amino_acid(prev_resname):
                        protein_part.append(line)
                    else:
                        ligand_part.append(line)
                elif line.startswith(("SSBOND", "HELIX", "SHEET", "SEQRES")):
                    protein_part.append(line)
                elif line.startswith("CONECT"):
                    # NOTE:
                    # Normally, CONECT records follow ATOM or HETATOM record parts.
                    atomids = [int(id_) for id_ in line[6:].split()]
                    if True in [id_ in ligand_atom_id for id_ in atomids]:
                        ligand_part.append(line)
                    else:
                        protein_part.append(line)

        self.receptor_pdbname = f"{self.pdbname}_rec.pdb"
        protein_file = open(self.receptor_pdbname, "w")
        for line in protein_part:
            protein_file.write(line)
        protein_file.close()

        if ligand_part:
            self.ligand_pdbname = f"{self.pdbname}_lig.pdb"
            ligand_file = open(self.ligand_pdbname, "w")
            for line in ligand_part:
                ligand_file.write(line)
            ligand_file.close()

    def load_protein_ligand(
        self,
        *,
        preserve_intermediates: bool = False,
        verbose: bool = False,
        desolvate: bool = False,
    ) -> None:
        # Loading protein pdb
        if not Path(self.receptor_pdbname).exists():
            raise FileNotFoundError(f"{self.receptor_pdbname} does not exist")
        self.receptor_obj = protein.Protein(self.receptor_pdbname, remove_wat=desolvate)
        if not preserve_intermediates:
            Path(self.receptor_pdbname).unlink()

        if Path(self.ligand_pdbname).exists():
            self.ligand_obj = ligand.Ligand(self.ligand_pdbname)
            if not preserve_intermediates:
                Path(self.ligand_pdbname).unlink()
        else:
            self.ligand_obj = ligand.Ligand("")  # vacant ligand object
            if verbose:
                raise UserWarning(f"{self.ligand_pdbname} does not exist. No ligands")

    def get_non_amino_acid_ligands(self) -> int:
        """
        Return number of non-water & non-atmino acid molecules.
        """
        return self.n_ligand

    def get_num_waters(self) -> int:
        """
        Return number of water molecules.
        """
        return self.n_water

    def search_ss_bond(self, *, ssbond_dist: float = 2.05):
        """
        Search SS bond in receptor protein object.
        """
        found: bool = False
        found = self.receptor_obj.search_ss_bond(ssbond_dist=ssbond_dist)
        return found

    def get_receptor_part_crds(self, *, chainid: str | list | None = None):
        """
        Return crd list (or npt.NDArray) of receptor parts.
        """
        return self.receptor_obj.get_crds_list(chainid=chainid)

    def get_ligand_part_crds(self, *, confid: int = 0):
        """
        Return crd list (= npt.NDArray) of ligand parts.
        """
        return self.ligand_obj.get_atomic_positions(conf_id=confid)

    def find_protein_ligand_contact_atom_pairs(
        self,
        *,
        dist_cut: float = 5.0,
        distant_fold: float = 3.0,
        as_numpy_array: bool = False,
    ) -> None:
        """
        Simply atomic contacts by finding contacts between heavy atoms of protein and ligands in terms of dist_cut.

        Note:
            - Store contacted atoms' id pairs in "self.atom_contact_pairs".
            - The atom indices in the atom_contact_pairs are "serial numbers" written in pdb files.
              (Not like array indices that start from 0 as used in programming.)

        Args:
            dist_cut: distance threshold for contact atomic pairs
            distant_fold: fold to detect the far distant residue which need not to be checked.
            as_num_array: convert atomic pairs into numpy format
        """
        self.atom_contact_pairs: list | np.ndarray = []

        receptor_all_residues = self.receptor_obj.get_residues()

        if self.n_ligand != 0:
            for latm_id in range(self.ligand_obj.get_atom_number()):
                latm = self.ligand_obj.get_ith_atom_obj(atom_id=latm_id)  # type: ignore
                if latm.GetSymbol() == "H":
                    # if ligand atom is hydrogen, skip this
                    continue
                latm_crd = self.ligand_obj.get_ith_atom_crd(atom_id=latm_id)
                for chid, chain_residues in receptor_all_residues.items():
                    # NOTE: about each chains
                    for resid, residue in chain_residues.items():
                        # NOTE: get residues sequentially
                        res_atoms = [atoms.copy() for atoms in residue.get_atoms()]
                        for patm in res_atoms:
                            if patm.get_name()[0] == "H":
                                # if protein atom is a hydrogen, skip this.
                                continue
                            patm_crd = patm.get_coord()
                            inter_atom_dist = np.linalg.norm(latm_crd - patm_crd)
                            # Check inter-atomic distance
                            if inter_atom_dist <= dist_cut:
                                # print(
                                #     latm_id,
                                #     latm.GetSymbol(),
                                #     self.ligand_obj.get_pdb_atomid(latm_id),
                                # )
                                self.atom_contact_pairs.append(
                                    (
                                        self.ligand_obj.get_pdb_atomid(latm_id),
                                        patm.get_serial_number(),
                                    )
                                )
                            elif inter_atom_dist > distant_fold * dist_cut:
                                # NOTE:
                                # If this atomic pair's inter-distance is larger than distnat_fold * dist-cut
                                # (in other words, this residue locates far from the ligand)
                                # skip this residue.
                                # -> for reducing calculation time
                                break
            if as_numpy_array:
                self.atom_contact_pairs = np.array(self.atom_contact_pairs)

    def find_active_site_residue(self, *, dist_cut: float = 6.0) -> list[bool]:
        """
        Find active site residues surrounding bound ligands.

        Args:
            dist_cut: distance threshoold for contacting residues.
        """
        receptor_all_residues = self.receptor_obj.get_residues()

        if self.n_ligand != 0:
            pres_id = 0
            for _, chain_residues in receptor_all_residues.items():
                # NOTE: about each chains
                for _, residue in chain_residues.items():
                    res_atoms = [atoms.copy() for atoms in residue.get_atoms()]
                    this_residue_contacted = False
                    for patm in res_atoms:
                        patm_crd = patm.get_coord()
                        for latm_id in range(self.ligand_obj.get_atom_number()):
                            # latm = self.ligand_obj.get_ith_atom_obj(atom_id=latm_id)
                            latm_crd = self.ligand_obj.get_ith_atom_crd(atom_id=latm_id)
                            # NOTE: pres_id : positional index to check out
                            #                 whether this residues are contacted with ligands

                            inter_atom_dist = np.linalg.norm(latm_crd - patm_crd)
                            # check inter-atomic distance
                            if inter_atom_dist <= dist_cut:
                                this_residue_contacted = True
                                break
                        if this_residue_contacted:
                            break
                    # NOTE:
                    # Because initially self.is_selected_residues is all True,
                    # when only active site residues are selected, make contacted residue as False
                    if not this_residue_contacted:
                        self.is_selected_residues[pres_id] = False
                    pres_id += 1

        return self.is_selected_residues

    def get_atomic_contact_pairs(self) -> list[int] | np.ndarray:
        """
        Return atomic contact pairs between a protein and a ligand.
        """
        return self.atom_contact_pairs

    def get_atomic_contact_pairs_as_array_index(
        self, *, as_numpy_array: bool = False
    ) -> list[int] | np.ndarray:
        """
        Return atomic contact pairs that are represented as list or array indexing type.
        """
        updated_contact_pair = []
        for pair in self.atom_contact_pairs:
            new_pair = (
                pair[0] - self.first_residue_resid,
                pair[1] - self.first_residue_resid,
            )
            updated_contact_pair.append(new_pair)
        if as_numpy_array:
            updated_contact_pair = np.array(updated_contact_pair)

        return updated_contact_pair

    # TODO: Needs to be implemented.
    # how to represent H-bond ???
    def find_hydrogen_bond_pairs(
        self,
        *,
        dist_cut: float = 3.2,
        angle_cut: float = 150.0,
        angle_deviation: float = 30.0,
        as_numpy_array: bool = False,
    ):
        """
        Find hydrogen bond interactions in terms of geometry.

        Args:
            dist_cut: distance between H -- acceptor
            angle_cut: angle of donor-H --  acceptor
            angle_deviation: possible deviation of angle
            as_numpy_array: convert hydrogen bond into numpy format
        """
        return None

    # TODO: Needs to be implemented.
    # how to represent hydrophobic clustering ??? (contacted size?
    def find_hydrophobic_contacts(self, *, dist_buffer: float = 0.5):
        """
        Find hydrophobic clusters which are the nearest collection of non-polar atoms.

        Note:
            Hydrophobic contacts are detected when the inter-atomic distance is larger than
            sum of van der Waals radius + dist_buffer (default: 0.5)

        Args:
            dist_buffer:
        """
        return None


if __name__ == "__main__":
    # test code
    complex_name = "test.pdb"
    complex_obj = ProteinLigandComplex(inpdb=complex_name)
    # TODO: test ProteinLigandComplex class works properly
    complex_obj.find_protein_ligand_contact_atom_pairs()
    for pair in complex_obj.get_atomic_contact_pairs():
        print(pair)
