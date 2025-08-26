"""
Uses processproduct and krswaps modules from stereopostprocessing
to perform post processing stereo correction on RetroTide proposed PKS design.
"""
# pylint: disable=no-member, import-error, wrong-import-position
from typing import Optional, List
import json
import yaml
from rdkit import Chem
from rdkit.Chem import Draw
import bcs

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config("RetroTideV2/stereopostprocessing/config.yaml")

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

modify_bcs_starters_extenders(starter_codes = config["starter_codes"],
                              extender_codes = config["extender_codes"])

from stereopostprocessing import processproduct as pp
from stereopostprocessing import krswaps
from stereopostprocessing import fingerprints as fp
from stereopostprocessing import similarity as sm

def preprocessing(unbound_product: Chem.Mol, target_mol: Chem.Mol):
    """
    Map PKS product to the target and assess correspondence between each
    chiral center
    """
    target_mol, mcs_mapped_atoms_df = pp.check_mcs(unbound_product, target_mol)
    full_mapping_df = pp.full_mapping(mcs_mapped_atoms_df,
                                      pp.map_product_to_pks_modules(unbound_product))
    chiral_result = pp.check_chiral_centers(unbound_product,
                                            target_mol,
                                            full_mapping_df)
    return chiral_result, full_mapping_df, target_mol

def postprocessing(pks_features_updated, final_design, target_mol: Chem.Mol, full_mapping_df):
    """
    Use new KR types to reconstruct the initial PKS design and its corresponding product
    """
    final_design = krswaps.new_design(pks_features_updated)[1]
    mapped_final_prod = pp.module_mapping(final_design)
    final_prod = pp.offload_pks_product(mapped_final_prod, target_mol, config["offload_mech"])[0]
    chiral_result_f = pp.check_chiral_centers(final_prod, target_mol, full_mapping_df)
    return chiral_result_f, final_prod, final_design

def compute_similarity(final_prod, target_mol):
    """
    Compute the jaccard similarity between the PKS product post KR Swaps and the target
    """
    fp_prod = fp.get_fingerprint(Chem.MolToSmiles(final_prod), 'mapchiral')
    fp_target = fp.get_fingerprint(Chem.MolToSmiles(target_mol), 'mapchiral')
    jaccard_score = sm.get_similarity(fp_prod, fp_target, 'jaccard')
    return jaccard_score

def plot_comparison(mol1: Chem.Mol, mol2: Chem.Mol, chiral_result):
    """
    Visualize results
    """
    highlight_1 = {}
    highlight_2 = {}
    # Green for matching chirality
    for atom_idx in chiral_result.match1:
        highlight_1[atom_idx] = (0, 1, 0)
    for atom_idx in chiral_result.match2:
        highlight_2[atom_idx] = (0, 1, 0)
    # Red for mismatching chirality
    for atom_idx in chiral_result.mmatch1:
        highlight_1[atom_idx] = (1, 0, 0)
    for atom_idx in chiral_result.mmatch2:
        highlight_2[atom_idx] = (1, 0, 0)
    # All highlighted atoms
    all_1 = chiral_result.match1 + chiral_result.mmatch1
    all_2 = chiral_result.match2 + chiral_result.mmatch2
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
    target_mol, pks_design, unbound_product = pp.initialize_pks_product(
        pp.canonicalize_smiles(molecule, config["stereo"]), config["offload_mech"])
    chiral_result, full_mapping_df, target_mol = preprocessing(unbound_product, target_mol)

    pks_features = krswaps.get_bcs_info(pks_design)
    pks_features_updated = krswaps.apply_kr_swaps(unbound_product, full_mapping_df,
                                                  chiral_result.mmatch1, pks_features)
    
    chiral_result_f, final_prod, final_design = postprocessing(pks_features_updated,
                                                               pks_design,
                                                               target_mol,
                                                               full_mapping_df)
    
    swaps_score = krswaps.check_swaps_accuracy(chiral_result_f.match1, chiral_result_f.mmatch1)
    if swaps_score is not None and swaps_score < 1.0:
        print("Remaining mismatches cannot be corrected by changing the KR type")
    if Chem.MolToSmiles(final_prod) == Chem.MolToSmiles(target_mol):
        print("The final product and matching target substructure smiles match!")
    else:
        print("Final product does NOT match the matching target substructure.")
        
    mol1 = pp.add_atom_labels(unbound_product, chiral_result.cc1)
    mol2 = pp.add_atom_labels(target_mol, chiral_result.cc2)
    mol1_f = pp.add_atom_labels(final_prod, chiral_result_f.cc1)
    mol2_f = pp.add_atom_labels(target_mol, chiral_result_f.cc2)
    return {
        "img_before": plot_comparison(mol1, mol2, chiral_result),
        "img_after": plot_comparison(mol1_f, mol2_f, chiral_result_f),
        "final_design": final_design,
        "jaccard": compute_similarity(final_prod, target_mol)
    }

if __name__ == "__main__":
    JOB_NAME = config["job_name"]
    MOLECULE = config["molecule"]
    OUTPUT_DIR = config["output_dir"]

    results = main(MOLECULE)
    # Output results
    with open(f'{OUTPUT_DIR}/{JOB_NAME}_final_design.json', 'w', encoding='utf-8') as json_file:
        json.dump([str(mod) for mod in results["final_design"]], json_file, indent=2)
    with open(f'{OUTPUT_DIR}/{JOB_NAME}_stereo_pre.svg', 'w', encoding='utf-8') as pre_img:
        pre_img.write(results["img_before"])
    with open(f'{OUTPUT_DIR}/{JOB_NAME}_stereo_post.svg', 'w', encoding='utf-8') as post_img:
        post_img.write(results["img_after"])
