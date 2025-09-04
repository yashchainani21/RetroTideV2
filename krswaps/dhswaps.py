from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, Draw
import pandas as pd
from typing import Optional, List
import json
import yaml
import bcs

def extract_ez_atoms(mol):
    """
    Returns lists of tuples for each (Z) and (E) double bond.
    """
    z_bonds = []
    e_bonds = []
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if stereo == Chem.BondStereo.STEREOZ:
                z_bonds.append((begin_idx, end_idx))
            elif stereo == Chem.BondStereo.STEREOE:
                e_bonds.append((begin_idx, end_idx))
    return z_bonds, e_bonds

def track_doublebond_info(target_mol: Chem.Mol):
    """
    """
    targ_z, targ_e = extract_ez_atoms(target_mol)
    z_set = set(idx for pair in targ_z for idx in pair)
    e_set = set(idx for pair in targ_e for idx in pair)
    return z_set, e_set

def ez_atom_labels(target_mol: Chem.Mol, full_mapping_df: pd.DataFrame):
    """
    """
    z_set, e_set = track_doublebond_info(target_mol)
    def get_bond_info(idx: int, target_mol: Chem.Mol):
        if idx in z_set:
            atom = target_mol.GetAtomWithIdx(idx)
            is_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())
            if is_double_bond:
                return "Z"
        if idx in e_set:
            atom = target_mol.GetAtomWithIdx(idx)
            is_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())
            if is_double_bond:
                return "E"
        else:
            return None
    
    full_mapping_df['EZ Label'] = full_mapping_df['Target Atom Idx'].apply(lambda idx: get_bond_info(idx, target_mol))
    return full_mapping_df

def target_module_dh(full_mapping_df: pd.DataFrame, z_bonds, e_bonds):
    z_targets = []
    for idx1, idx2 in z_bonds:
        mod1 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx1, 'Module'].values
        mod2 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx2, 'Module'].values
        z_targets.append((mod1[0], mod2[0]))
    e_targets = []
    for (idx1, idx2) in e_bonds:
        mod1 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx1, 'Module'].values
        mod2 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx2, 'Module'].values
        e_targets.append((mod1[0], mod2[0]))
    return z_targets, e_targets

def get_target_module_dh(double_bond: tuple):
    modi = (str(double_bond[0])).lstrip('M')
    modj = (str(double_bond[1])).lstrip('M')
    return int(modi) if modi > modj else int(modj)

def get_target_dh_modules(full_mapping_df: pd.DataFrame, target_mol: Chem.Mol):
    """
    """
    z_bonds, e_bonds = extract_ez_atoms(target_mol)
    z_target_mods, e_target_mods = target_module_dh(full_mapping_df, z_bonds, e_bonds)
    z_mod_num = [get_target_module_dh(db) for db in z_target_mods]
    e_mod_num = [get_target_module_dh(db) for db in e_target_mods]
    return z_mod_num, e_mod_num

def z_kr_dh_combo(target_z_module: list):
    new_kr_type = 'A'
    new_dh_type = 'Z'
    return new_kr_type, new_dh_type

def e_kr_dh_combo(target_e_module: list):
    new_kr_type = 'B'
    new_dh_type = 'E'
    return new_kr_type, new_dh_type

def check_substrate_ez_compatibility(pks_features: dict, z_mod_num: List[int]):
    """
    If the module number in pks_features equals the module number in full_mapping_df that has a Z EZ Label with a non-Malonyl substrate, then the Z bond is not possible via PKSs!!
    """
    error_ct = 0
    substrate = pks_features['Substrate']
    for mod in z_mod_num:
        if substrate[mod] != 'Malonyl-CoA':
            error_ct += 1
            print(f"Module {mod} is associated with an intended Z double bond but uses a non-Malonyl-CoA substrate ({substrate[mod]}).")
            print("------This transformation is incompatible with known PKS biosynthesis------")
    return error_ct

def check_dh_swaps_accuracy(error_ct: int):
    """
    # unmatching double bond EZ labels + # incompatible errors
    """
    #ADD EZ label mismatches
    return error_ct

def correct_ez_stereo(pks_features: dict, z_mod_num: list, e_mod_num: list) -> dict:
    for mod_z in z_mod_num:
        substrate = pks_features['Substrate'][mod_z]
        old_kr_type = pks_features['KR Type'][mod_z]
        old_dh_type = pks_features['DH Type'][mod_z]
        if old_kr_type is not None and old_dh_type is not None:
            if substrate == 'Malonyl-CoA':
                new_kr_type, new_dh_type = z_kr_dh_combo(mod_z)
                if new_kr_type != old_kr_type:
                    pks_features['KR Type'][mod_z] = new_kr_type
                    pks_features['DH Type'][mod_z] = new_dh_type
                    print(f"------Updating M{mod_z}------")
                    print(f"Swapped KR-DH types to types {new_kr_type}-{new_dh_type}")
    for mod_e in e_mod_num:
        substrate = pks_features['Substrate'][mod_e]
        old_kr_type = pks_features['KR Type'][mod_e]
        old_dh_type = pks_features['DH Type'][mod_e]
        if old_kr_type is not None and old_dh_type is not None:
            if substrate == 'Malonyl-CoA':
                new_kr_type, new_dh_type = e_kr_dh_combo(mod_e)
                if new_kr_type != old_kr_type:
                    pks_features['KR Type'][mod_e] = new_kr_type
                    pks_features['DH Type'][mod_e] = new_dh_type
                    print(f"------Updating M{mod_e}------")
                    print(f"Swapped KR-DH types to types {new_kr_type}-{new_dh_type}")
    return pks_features

def apply_dh_swaps(pks_features: dict, full_mapping_df: pd.DataFrame, target_mol: Chem.Mol) -> dict:
    """
    """
    z_mod_num, e_mod_num = get_target_dh_modules(full_mapping_df, target_mol)
    check_substrate_ez_compatibility(pks_features, z_mod_num)
    dh_swapped_pks_features = correct_ez_stereo(pks_features, z_mod_num, e_mod_num)
    return dh_swapped_pks_features