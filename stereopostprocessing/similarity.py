from rdkit import Chem
from rdkit.Chem import DataStructs, rdFMCS
from mapchiral.mapchiral import jaccard_similarity

def get_similarity(fp_1, fp_2, similarity_type: str):
    if similarity_type == "dice":
        score = DataStructs.DiceSimilarity(fp_1, fp_2)
        return score
    if similarity_type == "cosine":
        score = DataStructs.CosineSimilarity(fp_1, fp_2)
        return score
    if similarity_type == "tanimoto":
        score = DataStructs.TanimotoSimilarity(fp_1, fp_2)
        return score
    if similarity_type == "jaccard":
        score = jaccard_similarity(fp_1, fp_2)
        return score
    else:
        print("Invalid similarity type")

def num_atoms(true_product: str, pks_product: str, similarity_type: str):
    if similarity_type == "mcs-stereo-num-atoms":
        mol_1 = Chem.MolFromSmiles(true_product)
        mol_2 = Chem.MolFromSmiles(pks_product)
        atoms_1 = mol_1.GetNumAtoms()
        atoms_2 = mol_2.GetNumAtoms()
        result = rdFMCS.FindMCS([mol_1, mol_2], timeout=1, matchValences=True, matchChiralTag=True,
                                bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact)
        num_atoms = result.numAtoms
        score = num_atoms / (atoms_1 + atoms_2 - num_atoms)
        return score