from pathlib import Path

from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem
from rdkit.Chem import rdDistGeom  # type: ignore


class Ligand:
    """
    Ligand class.

    --------
    Inputs:
    ligand file (pdb, sdf, mol, mol2 files)
    """

    def __init__(self, inputfile_path):
        self.infile = inputfile_path
        self.mol: Chem.rdchem.Mol
        file_ext = Path(inputfile_path).suffix[1:]
        self.pdb_atomid: list[int] = []

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
        Chem.SanitizeMol(mol)  # type: ignore
        if add_h:
            mol = Chem.AddHs(mol)  # type: ignore
        # self.mol : main objects
        self.mol = Chem.Mol(mol)  # type: ignore

        # for getting atomic index written in pdb
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("ligand", infile)
        model = structure[0]  # type: ignore
        for chain in model:
            for residue in chain:
                for atom in residue:
                    self.pdb_atomid.append(atom.get_serial_number())

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

    # NOTE:
    # Geometric features
    def find_rotatable_bonds(self):
        self.rotatable_bond = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")  # type: ignore
        )
