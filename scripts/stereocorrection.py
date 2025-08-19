import bcs
from typing import Optional, List
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from itertools import product
from stereopostprocessing import fingerprints, similarity
import matplotlib.pyplot as plt
import time

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    '''
    Modifies the starter and extender units available for RetroTide.
    Removes all starter and extender units not specifed in the input lists.
    '''
    for key in list(bcs.starters.keys()):
        if key not in starter_codes:
            bcs.starters.pop(key, None)
            
    for key in list(bcs.extenders.keys()):
        if key not in extender_codes:
            bcs.extenders.pop(key, None)
    return

def canonicalize(molecule_str: str) -> str:
    '''
    Ensures the input molecule string is in canonical SMILES format.

    Returns:
        target (str): The canonical SMILES string of the molecule.
    '''
    mol = Chem.MolFromSmiles(molecule_str)
    target = Chem.MolToSmiles(mol)
    return target

def _pks_release_reaction(pks_release_mechanism: str, bound_product_mol: Chem.Mol) -> Chem.Mol:
    '''
    Runs a PKS offloading reaction to release the bound product from the PKS.

    Args:
        pks_release_mechanism (str): Mechanism to use for offloading ('thiolysis' or 'cyclization').
        bound_product_mol (Chem.Mol): The bound product molecule
    '''
    if pks_release_mechanism == 'thiolysis':
        Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to produce terminal acid group
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
        unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
        Chem.SanitizeMol(unbound_product_mol)
        return unbound_product_mol

    if pks_release_mechanism == 'cyclization':
        Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to cyclize bound substrate
        rxn = AllChem.ReactionFromSmarts('([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]')
        try:
            unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
            Chem.SanitizeMol(unbound_product_mol)
            return unbound_product_mol

        except:
            raise ValueError("\nUnable to perform cyclization reaction")

def initial_pks(target: str):
    '''
    Computes a PKS design to synthesize the 2D structure of the target molecule.

    Args:
        target (str): The SMILES string of the target molecule.

    Returns:
        pks_design (bcs object): Initial RetroTide design.
        mol (Chem.Mol): The computed product of the PKS design.
    '''
    designs = designPKS(Chem.MolFromSmiles(target),
                        maxDesignsPerRound = 200,
                        similarity = 'mcs_without_stereo') #Match target 2D structure
    pks_design = designs[-1][0][0].modules
    mol = designs[-1][0][0].computeProduct(structureDB)
    print('Score: ' + str(designs[-1][0][1]))
    print('Initial PKS Design: ' + str(pks_design))
    return pks_design, mol

def get_bcs_info(pks_design)-> dict:
    '''
    Constructs a dictionary containing information about the PKS design, including module number,
    substrate, and possible KR types.

    Args:
        pks_design (bcs object): Initial RetroTide design

    Returns:
        pks_features (dict): Dictionary with information about the intial PKS design
    '''
    pks_features = {
        'Module Number': [],
        'Substrate': [],
        'Possible KR Types': []
    }

    mal_KR_types = ['A', 'B']
    KR_types = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

    for idx, module in enumerate(pks_design):
        pks_features['Module Number'].append(idx)
        
        substrate = module.domains[bcs.AT].substrate
        pks_features['Substrate'].append(substrate)

        if bcs.KR in pks_design[idx].domains and bcs.DH not in pks_design[idx].domains:
            if substrate == 'Malonyl-CoA':
                types = mal_KR_types
                pks_features['Possible KR Types'].append(types)
            else:
                types = KR_types
                pks_features['Possible KR Types'].append(types)
        elif bcs.DH in pks_design[idx].domains:
            if substrate == 'Malonyl-CoA':
                types = ['B']
                pks_features['Possible KR Types'].append(types)
            else:
                types = ['B1']
                pks_features['Possible KR Types'].append(types)
        elif bcs.ER in pks_design[idx].domains: #redundant with elif DH case above (inconsistent w/ paper. CHECK)
            if substrate == 'Malonyl-CoA':
                types = ['B']
                pks_features['Possible KR Types'].append(types)
    return pks_features

def kr_swaps(design: dict, pks_design) -> list:
    '''
    Enumerates over all possible combinations of KR domains present in the PKS design, creating
    a bcs object for each combination and computing the resulting PKS product.

    Args:
        design (dict): Dictionary containing information about the PKS design.
        pks_design (bcs object): Initial RetroTide design
    
    Returns:
        pks_prod_str (list): List of SMILES strings of the PKS products for each KR combination.
        kr_combos_str (list): List of strings representing the KR combinations.
    '''
    modules = []
    pks_prod_str = []
    kr_types = design['Possible KR Types']
    kr_combos = list(product(*kr_types))
    kr_combos_str = ['-'.join(map(str, combo)) for combo in kr_combos]

    for kr_combo in kr_combos:
        modules = []
        num_kr_swaps_performed = 0
        for idx, substrate in enumerate(design['Substrate']):
            substrate = design['Substrate'][idx]
            if idx == 0:
                domains_dict = OrderedDict({bcs.AT: bcs.AT(active = True, substrate = substrate)})
                module = bcs.Module(domains = domains_dict, loading = True)
                modules.append(module)
            else:
                AT = bcs.AT(active = True, substrate = substrate)
                domains_dict = OrderedDict({bcs.AT: AT})
                
                if bcs.KR in pks_design[idx].domains:
                    domains_dict.update({bcs.KR: bcs.KR(active = True, type = kr_combo[num_kr_swaps_performed])})
                    num_kr_swaps_performed += 1
                if bcs.DH in pks_design[idx].domains:
                    domains_dict.update({bcs.DH: bcs.DH(active = True)})
                if bcs.ER in pks_design[idx].domains:
                    domains_dict.update({bcs.ER: bcs.ER(active = True)})

                module = bcs.Module(domains = domains_dict, loading = False)
                modules.append(module)
        cluster = bcs.Cluster(modules = modules)
        mol = cluster.computeProduct(structureDB)
        pks_product = Chem.MolToSmiles(mol)
        pks_prod_str.append(pks_product)
    return pks_prod_str, kr_combos_str

def release_pks_products(pks_products: list, pks_release_mechanism: str) -> list:
    '''
    Offloads the PKS products via thiolysis or cyclization.

    Args:
        pks_products (list): List of SMILES strings of bound PKS products.
        pks_release_mechanism (str): Mechanism to use for offloading ('thiolysis' or 'cyclization').
    
    Returns:
        released_smiles (list): List of SMILES strings of the released PKS products.
    '''
    released_smiles = []
    for smiles in pks_products:
        mol = Chem.MolFromSmiles(smiles)
        try:
            released_mol = _pks_release_reaction(pks_release_mechanism, mol)
        except Exception as e:
            if str(e) == "\nUnable to perform cyclization reaction":
                released_mol = _pks_release_reaction('thiolysis', mol)
        released_smiles.append(canonicalize(Chem.MolToSmiles(released_mol)))
    return released_smiles

def get_similarity_score(target_product:str, pks_products:str)-> list:
    '''
    Creates MAPC fingerprints for the target and PKS products. It then computes the Jaccard similarity
    score between the target and each PKS product.

    MAPC: Min-hashed Atom Pair Chiral Fingerprints (https://github.com/reymond-group/mapchiral)

    Args:
        target_product (str): SMILES string of the target molecule.
        pks_products (list): List of SMILES strings of the PKS products.
    
    Returns:
        scores (list): List of Jaccard similarity scores
    '''
    scores = []
    for product in pks_products:
        fp_1 = fingerprints.get_fingerprint(target_product, 'mapchiral')
        fp_2 = fingerprints.get_fingerprint(product, 'mapchiral')
        score = similarity.get_similarity(fp_1, fp_2, 'jaccard')
        scores.append(score)
    return scores

def plot_top_scores(kr_combos: list, scores: list, top_n: int):
    '''
    Plots the top N KR combinations in descending order by similarity score using a bar chart.

    Args:
        kr_combos (list): List of KR combinations.
        scores (list): List of similarity scores corresponding to each combo.
        top_n (int): Number of top combinations to plot.
    '''
    score_combo_pairs = list(zip(scores, kr_combos))
    score_combo_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Take top N
    top_pairs = score_combo_pairs[:top_n]
    top_scores = [pair[0] for pair in top_pairs]
    top_combos = [pair[1] for pair in top_pairs]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_combos)), top_scores, color='skyblue', edgecolor='black')
    plt.xlabel('KR Types (Top {} by Similarity Score)'.format(top_n))
    plt.ylabel('Mapchiral-Jaccard Similarity Score')
    plt.ylim(0, 1)
    plt.xticks(range(len(top_combos)), top_combos, rotation=45, ha='right', fontsize=8)
    
    for idx, score in enumerate(top_scores):
        plt.text(idx, score + 0.01, f"{score:.8f}", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

#OPTIONAL:
#TODO: Specify the starter and extender units to use in RetroTide.
modify_bcs_starters_extenders(starter_codes = ['trans-1,2-CPDA'],
                              extender_codes = ['Malonyl-CoA', 'Methylmalonyl-CoA'])
from retrotide import structureDB, designPKS

def main(molecule_str: str, offloading_mechanism: str) -> str:
    '''
    This function runs RetroTide to compute an initial PKS design to synthesize the 2D structure
    of the target molecule. It then enumerates over all possible combinations of KR domains
    present, creating a bcs object, and computes a similarity scores between the resulting products
    and the target molecule.

    Args:
        molecule_str (str): The SMILES string of the target molecule.
        offloading_mechanism (str): Mechanism to use for offloading ('thiolysis' or 'cyclization').

    Returns:
        best_kr_combo (str): The KR combination that yields the best PKS design.
    '''
    #Standardize the target SMILES string
    target = canonicalize(molecule_str)

    #Run RetroTide to get the initial PKS design
    initial_pks_design = initial_pks(target)

    #Store substrate and possible KR types information
    pks_features = get_bcs_info(initial_pks_design[0])

    #Perform KR swaps
    start = time.perf_counter()
    new_pks_design = kr_swaps(pks_features, initial_pks_design[0])
    end = time.perf_counter()
    print(f"Time taken for KR swaps: {end - start:.5f} seconds")

    #Offload the PKS products via thiolysis or cyclization
    released_products = release_pks_products(new_pks_design[0], offloading_mechanism)

    #Compute Jaccard similarity using MAPC fingerprints
    start_2 = time.perf_counter()
    similarity_scores = get_similarity_score(target, released_products)
    end_2 = time.perf_counter()
    print(f"Time taken for similarity calculation: {end_2 - start_2:.5f} seconds")

    #Visualize the top N KR combinations by similarity score
    plot_top_scores(new_pks_design[1], similarity_scores, top_n=10)

        # Create list of (score, KR_combo, SMILES) and sort by score
    results = list(zip(similarity_scores, new_pks_design[1], released_products))
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Get top 5 results
    top_5_results = [(combo, smiles, score) for score, combo, smiles in results[:5]]
    
    print("\nTop 5 KR combinations:")
    for i, (combo, smiles, score) in enumerate(top_5_results, 1):
        print(f"{i}. KR combo: {combo}")
        print(f"   Score: {score:.6f}")
        print(f"   SMILES: {smiles}")
        print()

        # Calculate difference between KR combo 3 and 4 scores
    if len(top_5_results) >= 4:
        score_3 = top_5_results[2][2]  # Index 2 for 3rd result, index 2 for score
        score_4 = top_5_results[3][2]  # Index 3 for 4th result, index 2 for score
        score_difference = (score_3 - score_4) * 1000000
        print(f"Score difference between KR combo 3 and 4: {score_difference:.2f} (×10⁶)")
    else:
        print("Not enough results to compare KR combo 3 and 4")

    #Output the KR combination leading to the PKS that best matches the R/S
    #stereochemistry of the target molecule
    max_score_idx = similarity_scores.index(max(similarity_scores))
    best_kr_combo = new_pks_design[1][max_score_idx]
    max_score = similarity_scores[max_score_idx]

    print(f"Best KR combination: {best_kr_combo} with score: {max_score:.3f}")
    return best_kr_combo

#TODO: Provide the SMILES string of the target molecule
molecule_str = 'C[C@H]1C[C@H](C[C@@H]([C@H](C(=CC=CC[C@H](OC(=O)C[C@@H]([C@H](C1)C)O)[C@@H]2CCC[C@H]2C(=O)O)C#N)O)C)C'

if __name__ == "__main__":
    overall_timer_start = time.perf_counter()
    new_design = main(molecule_str, 'cyclization')
    overall_timer_end = time.perf_counter()
    print(f"Overall processing time: {overall_timer_end - overall_timer_start:.4f} s")