from stereopostprocessing import similarity_scores
import numpy as np

#Unit Testing
def test_similarity_scores():
    propanol = similarity_scores.morganfp('OCCC(O)C')
    assert type(propanol) == np.ndarray
