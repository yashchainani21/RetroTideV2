from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from mapchiral.mapchiral import encode

def get_fingerprint(molecule_str: str, fp_type: str):
    mol = Chem.MolFromSmiles(molecule_str)
    if fp_type == "morgan_2D":
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize = 2048)
        fp = fpgen.GetFingerprint(mol)
        return fp
    if fp_type == "morgan_3D":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        return fp
    if fp_type == "mapchiral":
        fp = encode(mol, max_radius=2, n_permutations=2048, mapping=False)
        return fp
    else:
        print("Invalid fingerprinting method")