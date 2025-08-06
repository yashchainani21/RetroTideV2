"""
@author: Kenna Roberts
"""
# pylint: disable=no-member
import os
from typing import Optional, List
from rdkit import Chem
from rdkit.Chem import Draw
import bcs

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """
    Modifies the starter and extender units available for RetroTide.
    Removes all starter and extender units not specifed in the input lists.
    """
    for key in list(bcs.starters.keys()):
        if key not in starter_codes:
            bcs.starters.pop(key, None)
    for key in list(bcs.extenders.keys()):
        if key not in extender_codes:
            bcs.extenders.pop(key, None)

modify_bcs_starters_extenders(starter_codes = ['trans-1,2-CPDA'], extender_codes = ['Malonyl-CoA', 'Methylmalonyl-CoA'])

from retrotide import compareToTarget
from stereopostprocessing import processproduct as pp
from stereopostprocessing import krswaps
from stereopostprocessing import fingerprints as fp
from stereopostprocessing import similarity as sm

def add_atom_labels(mol: Chem.Mol, chiral_centers: dict) -> Chem.Mol:
    """
    Add atom labels to the molecule for visualization
    """
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom_idx = atom.GetIdx()
        base_label = f"{atom.GetSymbol()}:{atom_idx}"
        # Add R/S label if it's a chiral center
        if atom_idx in chiral_centers:
            chirality = chiral_centers[atom_idx]
            label = f"{base_label} ({chirality})"
        else:
            label = base_label
        atom.SetProp("atomNote", label)
    return mol_copy

def plot_comparison(mol1: Chem.Mol, mol2: Chem.Mol,
                     match1, match2, mmatch1, mmatch2):
    """
    Visualize results
    """
    highlight_1 = {}
    highlight_2 = {}
    # Green for matching chirality
    for atom_idx in match1:
        highlight_1[atom_idx] = (0, 1, 0)
    for atom_idx in match2:
        highlight_2[atom_idx] = (0, 1, 0)
    # Red for mismatching chirality
    for atom_idx in mmatch1:
        highlight_1[atom_idx] = (1, 0, 0)
    for atom_idx in mmatch2:
        highlight_2[atom_idx] = (1, 0, 0)
    # All highlighted atoms
    all_1 = match1 + mmatch1
    all_2 = match2 + mmatch2
    return Draw.MolsToGridImage([mol1, mol2], legends=['PKS Product', 'Target'], 
                                molsPerRow=2, highlightAtomLists=[all_1, all_2],
                                highlightAtomColors=[highlight_1, highlight_2],
                                highlightBondLists=[[], []], 
                                useSVG=True, 
                                subImgSize=(500, 400))

def main(molecule: str):
    """
    Perform post processing stereo correction on RetroTide proposed PKS design
    """
    target = pp.canonicalize_smiles(molecule, 'R/S')
    target_mol = Chem.MolFromSmiles(target)
    pks_design = pp.initial_pks(target)[0]
    pks_features = krswaps.get_bcs_info(pks_design)
    mapped_product = pp.module_mapping(pks_design)
    unbound_product = pp.offload_pks_product(mapped_product, target_mol, 'cyclization')[0]

    mcs_score = compareToTarget(unbound_product, target_mol, similarity='mcs_without_stereo')
    if mcs_score < 1.0:
        print(f"Initial PKS product only matches {mcs_score*100:.1f}% of the 2D target")
        print("Extracting common substructure from target to assess chiral centers from")
        mol2_match, mol2_copy = pp.matching_target_atoms(unbound_product, target_mol)
        mol2_submol = pp.extract_target_substructure(mol2_copy, list(mol2_match))

        mcs_mapped_atoms_df = pp.map_product_to_target(unbound_product, mol2_submol)
        target_mol = mol2_submol
    else:
        mcs_mapped_atoms_df = pp.map_product_to_target(unbound_product, target_mol)

    module_mapped_atoms_df = pp.map_product_to_pks_modules(unbound_product)
    full_mapping_df = pp.full_mapping(mcs_mapped_atoms_df, module_mapped_atoms_df)

    match1, match2, mmatch1, mmatch2, cc1, cc2 = pp.check_chiral_centers(unbound_product,
                                                                         target_mol,
                                                                         full_mapping_df)

    pairs = krswaps.find_adjacent_backbone_carbon_pairs(unbound_product, full_mapping_df)
    sequential_pairs = krswaps.filter_sequential_module_pairs(pairs)
    pairs_with_mismatches = krswaps.report_pairs_with_chiral_mismatches(sequential_pairs, mmatch1)
    pattern_results = krswaps.check_substituent_patterns(unbound_product, pairs_with_mismatches)
    pks_features_updated = krswaps.kr_swaps(pks_features, pattern_results, mmatch1)
    final_design = krswaps.new_design(pks_features_updated)[1]

    mapped_final_prod = pp.module_mapping(final_design)
    final_corrected_prod = pp.offload_pks_product(mapped_final_prod, target_mol, 'cyclization')[0]
    f_match1,f_match2,f_mmatch1,f_mmatch2,f_cc1,f_cc2 = pp.check_chiral_centers(final_corrected_prod,
                                                                                target_mol,
                                                                                full_mapping_df)

    if (len(f_match1) + len(f_mmatch1)) > 0:
        swaps_score = len(f_match1)/(len(f_match1)+len(f_mmatch1))
    else:
        swaps_score = "N/A"
    print(f"Swaps Score: {swaps_score:.3f}")

    final_product_smiles = Chem.MolToSmiles(final_corrected_prod)
    target_smiles = Chem.MolToSmiles(target_mol)

    if final_product_smiles == target_smiles:
        print("Final product matches the matching target substructure!")
    else:
        print("Final product does NOT match the matching target substructure.")
    
    fp_prod = fp.get_fingerprint(Chem.MolToSmiles(final_corrected_prod), 'mapchiral')
    fp_target = fp.get_fingerprint(target, 'mapchiral')
    jaccard_score = sm.get_similarity(fp_prod, fp_target, 'jaccard')
    print(f"Jaccard similarity between PKS product and the full target: {jaccard_score:.3f}")

    mol1 = add_atom_labels(unbound_product, cc1)
    mol2 = add_atom_labels(target_mol, cc2)

    mol1_f = add_atom_labels(final_corrected_prod, f_cc1)
    mol2_f = add_atom_labels(target_mol, f_cc2)

    img_before = plot_comparison(mol1, mol2, match1, match2, mmatch1, mmatch2)
    img_after = plot_comparison(mol1_f, mol2_f, f_match1, f_match2, f_mmatch1, f_mmatch2)
    return img_before, img_after, final_design

if __name__ == "__main__":
    JOB_NAME = 'Borrelidin' # Specify job name
    MOLECULE = 'C[C@H]1C[C@H](C[C@@H]([C@H](/C(=C\C=C\C[C@H](OC(=O)C[C@@H]([C@H](C1)C)O)[C@@H]2CCC[C@H]2C(=O)O)/C#N)O)C)C' # Insert target SMILES
    pre_comp, post_comp, pks_final_design = main(MOLECULE)

    print(f"PKS design after stereo correction: {pks_final_design}")

    OUTPUT_DIR = 'RetroTideV2/stereopostprocessing/results' # Specify path to output directory
    with open(f'{OUTPUT_DIR}/{JOB_NAME}_mcs_mapping_image_pre.svg', 'w', encoding='utf-8') as f:
        f.write(pre_comp)
    with open(f'{OUTPUT_DIR}/{JOB_NAME}_mcs_mapping_image_post.svg', 'w', encoding='utf-8') as f:
        f.write(post_comp)
