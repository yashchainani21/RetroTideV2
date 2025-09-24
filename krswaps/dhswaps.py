"""
Module to handle DH swaps based on alkene mismatches between the PKS product and target molecule
"""
# pylint: disable=no-member
from collections import namedtuple
from typing import List
from rdkit import Chem
import pandas as pd

def extract_ez_atoms(mol):
    """
    Identify atoms involved in Z and E alkenes present
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
    z_set = set(idx for pair in z_bonds for idx in pair)
    e_set = set(idx for pair in e_bonds for idx in pair)
    return z_set, e_set, z_bonds, e_bonds

def extract_ez_labels(pks_product: Chem.Mol, target_mol: Chem.Mol, full_mapping_df: pd.DataFrame):
    """
    Append Z and E labels to the full mapping dataframe
    """
    z_prod_set, e_prod_set, z_bonds, e_bonds = extract_ez_atoms(pks_product)
    z_targ_set, e_targ_set, z_targ_bonds, e_targ_bonds = extract_ez_atoms(target_mol)
    def get_bond_label(idx: int, mol: Chem.Mol, z_set, e_set):
        if idx in z_set:
            atom = mol.GetAtomWithIdx(idx)
            is_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())
            if is_double_bond:
                return "Z"
        if idx in e_set:
            atom = mol.GetAtomWithIdx(idx)
            is_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())
            if is_double_bond:
                return "E"
        return None
    full_mapping_df['Product EZ Label'] = full_mapping_df['Product Atom Idx'].apply(
        lambda idx: get_bond_label(idx, pks_product, z_prod_set, e_prod_set))
    full_mapping_df['Target EZ Label'] = full_mapping_df['Target Atom Idx'].apply(
        lambda idx: get_bond_label(idx, target_mol, z_targ_set, e_targ_set))
    return full_mapping_df

def alkene_mapped_modules(full_mapping_df: pd.DataFrame, z_bonds, e_bonds):
    """
    Map double bond atom indices to module numbers
    """
    z_targets = []
    for (idx1, idx2) in z_bonds:
        mod1 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx1, 'Module'].values
        mod2 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx2, 'Module'].values
        z_targets.append((mod1[0], mod2[0]))
    e_targets = []
    for (idx1, idx2) in e_bonds:
        mod1 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx1, 'Module'].values
        mod2 = full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx2, 'Module'].values
        e_targets.append((mod1[0], mod2[0]))
    return z_targets, e_targets

def pick_higher_module_dh(double_bond: tuple):
    """
    The target module is module Mi+1
    """
    modi = str(double_bond[0])
    modj = str(double_bond[1])
    if modi == 'LM':
        modi = 0
    else:
        modi = modi.lstrip('M')
    if modj == 'LM':
        modj = 0
    else:
        modj = modj.lstrip('M')
    return int(modi) if int(modi) > int(modj) else int(modj)

def identify_target_dh_modules(full_mapping_df: pd.DataFrame, target_mol: Chem.Mol):
    """
    Identify target modules associated with Z and E double bonds
    """
    z_set, e_set, z_bonds, e_bonds = extract_ez_atoms(target_mol)
    z_target_mods, e_target_mods = alkene_mapped_modules(full_mapping_df, z_bonds, e_bonds)
    z_mod_num = [pick_higher_module_dh(db) for db in z_target_mods]
    e_mod_num = [pick_higher_module_dh(db) for db in e_target_mods]
    return z_mod_num, e_mod_num

def z_kr_dh_combo(target_z_module: list):
    """
    KR-DH subtypes to produce Z double bonds
    """
    new_kr_type = 'A'
    new_dh_type = 'Z'
    return new_kr_type, new_dh_type

def e_kr_dh_combo(target_e_module: list):
    """
    KR-DH subtypes to produce E double bonds
    """
    new_kr_type = 'B'
    new_dh_type = 'E'
    return new_kr_type, new_dh_type

def check_substrate_ez_compatibility(pks_features: dict, z_mod_num: List[int]):
    """
    Checks if modules intended to produce Z double bonds use Malonyl-CoA as substrate.
    """
    error_ct = 0
    substrate = pks_features['Substrate']
    for mod in z_mod_num:
        if substrate[mod] != 'Malonyl-CoA':
            error_ct += 1
            print(f"Module {mod} is associated with an intended Z double bond but uses a non-Malonyl-CoA substrate ({substrate[mod]}).")
            print("------This transformation is incompatible with known PKS biosynthesis------")
    return error_ct

def correct_ez_stereo(pks_features: dict, z_mod_num: list, e_mod_num: list) -> dict:
    """
    Swap previous KR-DH subtypes to those that produce the desired alkene stereochemistry
    """
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
                    print(f"    Making KR-DH swap in module {mod_z}")
                    print(f"    M{mod_z}: Swapped KR-DH types to types {new_kr_type}-{new_dh_type}")
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
                    print(f"    Making KR-DH swap in module {mod_e}")
                    print(f"    M{mod_e}: Swapped KR-DH types to types {new_kr_type}-{new_dh_type}")
    return pks_features

def apply_dh_swaps(pks_features: dict, full_mapping_df: pd.DataFrame, target_mol: Chem.Mol) -> dict:
    """
    Make KR-DH subtype swaps to correct alkene stereochemistry
    """
    z_mod_num, e_mod_num = identify_target_dh_modules(full_mapping_df, target_mol)
    check_substrate_ez_compatibility(pks_features, z_mod_num)
    dh_swapped_pks_features = correct_ez_stereo(pks_features, z_mod_num, e_mod_num)
    return dh_swapped_pks_features

AlkeneCheckResult = namedtuple('AlkeneCheckResult', ['match1', 'match2', 'mmatch1', 'mmatch2'])
def check_alkene_stereo(mapped_atoms: pd.DataFrame) -> AlkeneCheckResult:
    """
    Checks alkene stereochemistry correspondence between the PKS product and target molecule

    Returns:
        AlkeneCheckResult: A named tuple containing lists of matching and mismatching atom indices
        and dicts of double bonds
    """
    matching_atoms_1 = []
    matching_atoms_2 = []
    mismatching_atoms_1 = []
    mismatching_atoms_2 = []
    mapped_atoms['EZ Match'] = mapped_atoms['Target EZ Label'] == mapped_atoms['Product EZ Label']
    mapped_atoms['EZ Mismatch'] = mapped_atoms['Target EZ Label'] != mapped_atoms['Product EZ Label']
    for _, row in mapped_atoms.iterrows():
        if row['EZ Match'] and row['Target EZ Label']:
            matching_atoms_1.append(row['Product Atom Idx'])
            matching_atoms_2.append(row['Target Atom Idx'])
        if row['EZ Mismatch'] and row['Target EZ Label']:
            mismatching_atoms_1.append(row['Product Atom Idx'])
            mismatching_atoms_2.append(row['Target Atom Idx'])
    return AlkeneCheckResult(matching_atoms_1, matching_atoms_2,
                             mismatching_atoms_1, mismatching_atoms_2)
