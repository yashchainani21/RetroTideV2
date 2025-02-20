from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem import DataStructs, rdFMCS

def morganfp(molecule_str: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(molecule_str)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    return np.array(fp)

