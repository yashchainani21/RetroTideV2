from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, Draw
import pandas as pd
from typing import Optional, List
import json
import yaml
import bcs

def extract_ez_atoms_full(mol):
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

def label_ez(idx, target_mol, z_bonds, e_bonds):
    z_set = set(idx for pair in z_bonds for idx in pair)
    e_set = set(idx for pair in e_bonds for idx in pair)
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
    
def update_ez_labels(full_mapping: pd.DataFrame, target_mol: Chem.Mol, z_bonds, e_bonds) -> pd.DataFrame:
    full_mapping['EZ Label'] = full_mapping['Target Atom Idx'].apply(lambda idx: label_ez(idx,
                                                                                          target_mol,
                                                                                          z_bonds,
                                                                                          e_bonds))
    return full_mapping

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

def z_kr_dh_combo(target_z_module: list):
    new_kr_type = 'A'
    new_dh_type = 'Z'
    return new_kr_type, new_dh_type

def e_kr_dh_combo(target_e_module: list):
    new_kr_type = 'B'
    new_dh_type = 'E'
    return new_kr_type, new_dh_type

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

