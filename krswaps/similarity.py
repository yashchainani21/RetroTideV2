from rdkit import Chem
from rdkit.Chem import DataStructs, rdFMCS
from mapchiral.mapchiral import jaccard_similarity

def get_similarity(fp_1, fp_2, similarity_type: str):
    """
    Compute the chemical similarity between two molecules represented
    by their fingerprints

    Args:
        fp_1: Fingerprint of molecule 1
        fp_2: Fingerprint of molecule 2
        similarity_type (str): Type of similarity metric to use
    """
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