import os
from stereopostprocessing import fingerprints
from stereopostprocessing import similarity
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

#Define fingerprinting methods and similarity metrics
fp_types = ["morgan_2D", "morgan_3D"]
similarity_types = ["dice", "cosine", "tanimoto"]

#Quantify the chemical similarity [0, 1] between two molecules
def get_score(true_product: str, pks_product: str) -> np.ndarray:
    scores = []
    fp_method = []
    sim_metric = []
    for fp in fp_types:
        for sim in similarity_types:
            fp_1 = fingerprints.get_fingerprint(true_product, fp)
            fp_2 = fingerprints.get_fingerprint(pks_product, fp)
            sim_score = similarity.get_similarity(fp_1, fp_2, sim)
            scores.append(sim_score)
            fp_method.append(fp)
            sim_metric.append(sim)
    result = np.column_stack((fp_method, sim_metric, scores))
    return result

def mapchiral_jaccard(true_product: str, pks_product: str) -> np.ndarray:
    scores = []
    fp_method = []
    sim_metric = []
    fp_1 = fingerprints.get_fingerprint(true_product, "mapchiral")
    fp_2 = fingerprints.get_fingerprint(pks_product, "mapchiral")
    sim_score = similarity.get_similarity(fp_1, fp_2, "jaccard")
    scores.append(sim_score)
    fp_method.append("mapchiral")
    sim_metric.append("jaccard")
    exception = np.column_stack((fp_method, sim_metric, scores))
    return exception

def check_num_atoms(true_product: str, pks_product: str) -> np.ndarray:
    scores = []
    fp_method = ["-"]
    sim_metric = []
    sim_score = similarity.num_atoms(true_product, pks_product, "mcs-stereo-num-atoms")
    scores.append(sim_score)
    sim_metric.append("mcs-stereo-num-atoms")
    exception_num_atoms = np.column_stack((fp_method, sim_metric, scores))
    return exception_num_atoms

#Read SMILE strings and stereocenter labels for the true product and its stereoisomers from a csv file
path = r"RetroTideV2/scripts/6deoxyerythronolideb_example.csv"
df = pd.read_csv(path)

stereoisomers = df['SMILE String']
stereolabels = df['Stereocenter Labels']
ground_truth = stereoisomers.iloc[0]
print(ground_truth)

#Enumerate over stereoisomers of the true product to obtain a similarity score
scores = []
for isomer in stereoisomers:
    result = get_score(ground_truth, isomer)
    scores.append(result)
    exception = mapchiral_jaccard(ground_truth, isomer)
    scores.append(exception)
    #exception_num_atoms = check_num_atoms(ground_truth, isomer)
    #scores.append(exception_num_atoms)

scores = np.concatenate(scores)
length = len(stereoisomers)
sim_scores = scores[:, 2].astype(float).reshape(length, -1)

#Plot a 2D heatmap
plt.figure(figsize=(15, 12))

xticklabels = [f"{fp}-{sim}" for fp, sim in zip(scores[:7, 0], scores[:7, 1])]
yticklabels = stereolabels

heatmap = sns.heatmap(sim_scores, xticklabels = xticklabels, yticklabels = yticklabels, cmap="YlGnBu", annot = True,
                      annot_kws={"size": 15}, fmt = '.3g')
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize = 15, rotation = 60)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize = 15)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
cbar.set_label("Similarity score", fontsize = 15)
plt.xlabel("Fingerprint/Similarity Combo", size = 15)
plt.ylabel("Stereocenter Configuration", size = 15)
plt.tight_layout()
plt.show()