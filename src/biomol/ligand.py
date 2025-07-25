from pathlib import Path
from typing import Literal

from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    rdchem,  # pyright: ignore[reportAttributeAccessIssue]
    rdDistGeom,  # pyright: ignore[reportAttributeAccessIssue]
    rdEHTTools,  # pyright: ignore[reportAttributeAccessIssue]
)
from rdkit.Chem.rdchem import BondType


class Ligand:
    """
    Ligand class.

    --------
    Inputs:
    ligand file (pdb, sdf, mol, mol2 files)
    """

    def __init__(self, inputfile_path: str):
        self.mol: Chem.rdchem.Mol
        self.infile = inputfile_path
        self.lig_name: str = ""
        self.pdb_atomid: list[int] = []
        self.atom_names: list[str] = []

        file_ext = Path(inputfile_path).suffix[1:]
        if file_ext == "pdb":
            self.loadpdb(infile=self.infile)
        else:
            # TODO:
            # Consider the cases of mol, sdf, and mol2.
            #
            pass

        # self.find_rotatable_bonds()

    def loadpdb(self, *, infile: str, add_h: bool = False):
        # 1. Load PDB with connectivity
        mol = Chem.MolFromPDBFile(infile, removeHs=False, sanitize=False)  # type: ignore
        # 2. Sanitize manually to assign bond order
        Chem.SanitizeMol(mol)  # pyright: ignore[reportAttributeAccessIssue]
        if add_h:
            mol = Chem.AddHs(mol)  # pyright: ignore[reportAttributeAccessIssue]
        # NOTE: self.mol : main objects
        self.mol = Chem.Mol(mol)  # pyright: ignore[reportAttributeAccessIssue]

        # for getting atomic index written in pdb
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("ligand", infile)
        model = structure[0]  # type: ignore
        for chain in model:
            for residue in chain:
                if self.lig_name == "":
                    self.lig_name = residue.get_resname()
                for atom in residue:
                    self.pdb_atomid.append(atom.get_serial_number())
                    self.atom_names.append(atom.get_name())
        # update ligand name as atoms' property
        for atom in mol.GetAtoms():
            atom.SetProp("_ligand_name", self.lig_name)

    def get_ligand_name(self):
        """
        Return ligand's name.
        """
        return self.lig_name

    @staticmethod
    def get_atoms_ligand_name(*, atom: rdchem.Atom):
        """
        Return atom's its belonging ligand name.
        """
        return atom.GetProp("_ligand_name")

    def get_pdb_atomid(self, id_: int) -> int:
        """
        Return an atomic id as this ligand is in the complex pdb.

        The ligand should be after the receptor part in the complex pdb.

        Input:
            id_: int, atom's order id in the mol object.
        Output:
            atomic index when this ligand complexed with a protein.

            If initially ligand.pdb is given, which is extracted from complex pdb,
            this will return the atomic serial number written in the pdb file.
        """
        return self.pdb_atomid[id_]

    def get_atom_number(self) -> int:
        """
        Return number of atoms in this ligand.
        """
        return self.mol.GetNumAtoms()

    def get_atoms(self) -> list:
        """
        Return list of rdkit atom objects of this ligand.
        """
        atom_list = [atom for atom in self.mol.GetAtoms()]
        return atom_list

    # FIX:
    #  - Not appropriately work when the ligand is given as pdb file without CONECT
    #    -> bond.GetBondType() not work properly.
    def calc_net_charge(self) -> float:
        """
        Calculate and update molecular net charge.
        """
        NEUTRAL_ATOM_CONNECT = {
            "O": [2],
            "N": [3],
            "C": [4],
            "F": [1],
            "Cl": [1],
            "H": [1],
            "S": [2, 4, 6],
        }
        net_charge = 0.0

        # Gasteiger charge
        gast_net_charge = 0.0
        AllChem.ComputeGasteigerCharges(self.mol)  # pyright: ignore[reportAttributeAccessIssue]
        for atom in self.mol.GetAtoms():
            gast_net_charge += atom.GetDoubleProp("_GasteigerCharge")
        gast_net_charge = round(gast_net_charge)

        mmff_net_charge = 0.0
        fps = AllChem.MMFFGetMoleculeProperties(self.mol)  # pyright: ignore[reportUnusedVariable, reportAttributeAccessIssue]
        for i in range(self.mol.GetNumAtoms()):
            mmff_net_charge += fps.GetMMFFPartialCharge(i)
        mmff_net_charge = round(mmff_net_charge)

        if gast_net_charge == mmff_net_charge:
            net_charge = mmff_net_charge
        else:
            # if Gasteiger charge != MMFF charge, do semiempirical QM charge
            rdDistGeom.EmbedMolecule(self.mol)
            ok, res = rdEHTTools.RunMol(self.mol)
            net_charge = round(sum(res.GetAtomicCharges()))

        return net_charge

    # NOTE:
    # 3D informations
    #################
    def get_conformer(self, *, conf_id=0):
        """
        Return "the copied" conformer.
        """
        if not self.mol.GetConformer().Is3D():
            # if current conformaer is 2D
            rdDistGeom.EmbedMolecule(self.mol)
        return Chem.Conformer(self.mol.GetConformer(conf_id))  # type: ignore

    def get_atomic_positions(self, *, conf_id: int = 0):
        """
        Return atomic coordinates of ith conformors (default: 0 = first conformers).
        """
        return self.mol.GetConformer(conf_id).GetPositions()

    def get_ith_atom_crd(self, *, atom_id: int, conf_id: int = 0):
        """
        Return ith atom's 3D coordinates.
        """
        return self.mol.GetConformer(conf_id).GetPositions()[atom_id]

    def get_ith_atom_obj(self, *, atom_id: int):
        """
        Return ith atom's rdkit object.
        """
        return self.mol.GetAtomWithIdx(atom_id)

    def get_ith_atom_element(self, *, atom_id: int) -> str:
        """
        Return ith atom's element symbol.
        """
        return self.mol.GetAtomWithIdx(atom_id).getSymbol()

    def get_ith_atom_name(self, *, atom_id: int) -> str:
        """
        Return ith atom's name written in the input file.
        """
        return self.atom_names[atom_id]

    # NOTE:
    # Geometric features
    def find_rotatable_bonds(self):
        self.rotatable_bond = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")  # type: ignore
        )
