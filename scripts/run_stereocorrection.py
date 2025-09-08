"""
Uses processproduct, krswaps, and dhswaps modules from stereopostprocessing
to perform post processing E/Z and R/S stereo corrections on RetroTide proposed PKS design.
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

config = load_config("RetroTideV2/krswaps/config.yaml")

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

from krswaps import processproduct as pp
from krswaps import krswaps, dhswaps
from krswaps import fingerprints as fp
from krswaps import similarity as sm

def preprocessing(unbound_product: Chem.Mol, target_mol: Chem.Mol):
    """
    Map PKS product to the target and assess correspondence between each
    chiral center
    """
    target_mol, mcs_mapped_atoms_df = pp.check_mcs(unbound_product, target_mol)
    full_mapping_df = dhswaps.ez_atom_labels(unbound_product, target_mol,
                                             pp.full_mapping(mcs_mapped_atoms_df,
                                                             pp.map_product_to_pks_modules(unbound_product)))
    chiral_result = pp.check_chiral_centers(unbound_product,
                                            target_mol,
                                            full_mapping_df)
    alkene_result = dhswaps.check_alkene_stereo(full_mapping_df)
    return chiral_result, alkene_result, full_mapping_df, target_mol

def postprocessing(pks_features_updated, final_design, target_mol: Chem.Mol, full_mapping_df):
    """
    Use new KR types to reconstruct the initial PKS design and its corresponding product
    """
    final_design = krswaps.new_design(pks_features_updated)[1]
    mapped_final_prod = pp.module_mapping(final_design)
    final_prod = pp.offload_pks_product(mapped_final_prod, target_mol, config["offload_mech"])[0]
    full_mapping_df_f = dhswaps.ez_atom_labels(final_prod, target_mol, full_mapping_df)
    chiral_result_f = pp.check_chiral_centers(final_prod, target_mol, full_mapping_df)
    alkene_result_f = dhswaps.check_alkene_stereo(full_mapping_df_f)
    return chiral_result_f, alkene_result_f, final_prod, final_design

def compute_similarity(final_prod, target_mol):
    """
    Compute the jaccard similarity between the PKS product post KR Swaps and the target
    """
    fp_prod = fp.get_fingerprint(Chem.MolToSmiles(final_prod), 'mapchiral')
    fp_target = fp.get_fingerprint(Chem.MolToSmiles(target_mol), 'mapchiral')
    jaccard_score = sm.get_similarity(fp_prod, fp_target, 'jaccard')
    return jaccard_score

def plot_stereo_comparison(mol1: Chem.Mol, mol2: Chem.Mol, chiral_result, alkene_result):
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

    # Green for matching alkene stereochemistry
    for atom_idx in alkene_result.match1:
        highlight_1[atom_idx] = (0, 1, 0)
    for atom_idx in alkene_result.match2:
        highlight_2[atom_idx] = (0, 1, 0)
    # Red for mismatching alkene stereochemistry
    for atom_idx in alkene_result.mmatch1:
        highlight_1[atom_idx] = (1, 0, 0)
    for atom_idx in alkene_result.mmatch2:
        highlight_2[atom_idx] = (1, 0, 0)

    # All highlighted atoms
    all_1 = chiral_result.match1 + chiral_result.mmatch1 + alkene_result.match1 + alkene_result.mmatch1
    all_2 = chiral_result.match2 + chiral_result.mmatch2 + alkene_result.match2 + alkene_result.mmatch2

    # All highlighted double bonds
    all_bonds_1 = set(alkene_result.match1 + alkene_result.mmatch1)
    all_bonds_2 = set(alkene_result.match2 + alkene_result.mmatch2)

    bond_indices_1 = [bond.GetIdx() for bond in mol1.GetBonds()
                      if bond.GetBeginAtomIdx() in all_bonds_1
                      and bond.GetEndAtomIdx() in all_bonds_1]
    bond_indices_2 = [bond.GetIdx() for bond in mol2.GetBonds()
                      if bond.GetBeginAtomIdx() in all_bonds_2
                      and bond.GetEndAtomIdx() in all_bonds_2]

    return Draw.MolsToGridImage([mol1, mol2], legends=['PKS Product', 'Target'], 
                                molsPerRow=2, highlightAtomLists=[all_1, all_2],
                                highlightAtomColors=[highlight_1, highlight_2],
                                highlightBondLists=[bond_indices_1, bond_indices_2], 
                                useSVG=True, 
                                subImgSize=(500, 400))

def main(molecule: str):
    """
    Perform post processing stereo correction on RetroTide proposed PKS design
    """
    target_mol, pks_design, unbound_product = pp.initialize_pks_product(
        pp.canonicalize_smiles(molecule, config["stereo"]), config["offload_mech"])
    chiral_result, alkene_result, full_mapping_df, target_mol = preprocessing(unbound_product, target_mol)

    pks_features = krswaps.get_bcs_info(pks_design)
    print("Correcting R/S Stereochemistry")
    pks_features_updated = krswaps.apply_kr_swaps(unbound_product, full_mapping_df,
                                                  chiral_result.mmatch1, pks_features)
    print("Correcting E/Z Stereochemistry")
    pks_features_dh_swapped = dhswaps.apply_dh_swaps(pks_features_updated, full_mapping_df, target_mol)

    chiral_result_f, alkene_result_f, final_prod, final_design = postprocessing(pks_features_dh_swapped,
                                                               pks_design,
                                                               target_mol,
                                                               full_mapping_df)
    
    rs_swaps_score = krswaps.check_swaps_accuracy(chiral_result_f.match1, chiral_result_f.mmatch1)
    #Add ez_swaps_score
    if rs_swaps_score is not None and rs_swaps_score < 1.0:
        print("Remaining R/S mismatches cannot be corrected by changing the KR type")
    mol1 = pp.add_atom_labels(unbound_product, chiral_result.cc1)
    mol2 = pp.add_atom_labels(target_mol, chiral_result.cc2)
    mol1_f = pp.add_atom_labels(final_prod, chiral_result_f.cc1)
    mol2_f = pp.add_atom_labels(target_mol, chiral_result_f.cc2)
    return {
        "img_before": plot_stereo_comparison(mol1, mol2, chiral_result, alkene_result),
        "img_after": plot_stereo_comparison(mol1_f, mol2_f, chiral_result_f, alkene_result_f),
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
        json.dump({
            "Final PKS Design":[str(mod) for mod in results["final_design"]],
            "Jaccard Similarity Score": results["jaccard"]
            }, json_file, indent=2)
    with open(f'{OUTPUT_DIR}/{JOB_NAME}_stereo_pre.svg', 'w', encoding='utf-8') as pre_img:
        pre_img.write(results["img_before"])
    with open(f'{OUTPUT_DIR}/{JOB_NAME}_stereo_post.svg', 'w', encoding='utf-8') as post_img:
        post_img.write(results["img_after"])
