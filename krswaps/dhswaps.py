"""
Module to handle DH swaps based on alkene mismatches between the PKS product and target molecule
"""
# pylint: disable=no-member
from collections import namedtuple
from typing import List, Tuple
from rdkit import Chem
import pandas as pd
from krswaps import krswaps

def extract_ez_atoms(mol: Chem.Mol) -> Tuple[list, list]:
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
    return z_bonds, e_bonds

def extract_ez_labels(pks_product: Chem.Mol, target_mol: Chem.Mol, full_mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    For atoms in Z and E alkenes present, append Z and E labels to the full mapping dataframe
    """
    def get_bond_label(idx: int, mol: Chem.Mol):
        z_bonds, e_bonds = extract_ez_atoms(mol)
        if idx in {i for bond in z_bonds for i in bond}:
            atom = mol.GetAtomWithIdx(idx)
            is_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())
            if is_double_bond:
                return "Z"
        if idx in {i for bond in e_bonds for i in bond}:
            atom = mol.GetAtomWithIdx(idx)
            is_double_bond = any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds())
            if is_double_bond:
                return "E"
        return None
    full_mapping_df['Product EZ Label'] = full_mapping_df['Product Atom Idx'].apply(
        lambda idx: get_bond_label(idx, pks_product))
    full_mapping_df['Target EZ Label'] = full_mapping_df['Target Atom Idx'].apply(
        lambda idx: get_bond_label(idx, target_mol))
    return full_mapping_df

def identify_dh_modules(full_mapping_df: pd.DataFrame, z_bonds: list, e_bonds: list) -> Tuple[list, list]:
    """
    Map double bond atom indices to module numbers. For each pair, identify Mi+1 as the
    target module to make KR-DH swap in

    Returns:
        z_targets (list): Modules facilitating Z alkene formation
        e_targets (list): Modules facilitating E alkene formation
    """
    z_targets = []
    for (idx1, idx2) in z_bonds:
        mod1 = krswaps.get_module_number(
            str(full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx1, 'Module'].values[0])
        )
        mod2 = krswaps.get_module_number(
            str(full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx2, 'Module'].values[0])
        )
        targ_mod = mod1 if mod1 > mod2 else mod2
        z_targets.append(targ_mod)
    e_targets = []
    for (idx1, idx2) in e_bonds:
        mod1 = krswaps.get_module_number(
            full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx1, 'Module'].values[0])
        mod2 = krswaps.get_module_number(
            full_mapping_df.loc[full_mapping_df['Target Atom Idx'] == idx2, 'Module'].values[0])
        targ_mod = mod1 if mod1 > mod2 else mod2
        e_targets.append(targ_mod)
    return z_targets, e_targets

def z_kr_dh_combo() -> Tuple[str, str]:
    """
    KR-DH subtypes to produce Z double bonds
    """
    new_kr_type = 'A'
    new_dh_type = 'Z'
    return new_kr_type, new_dh_type

def e_kr_dh_combo() -> Tuple[str, str]:
    """
    KR-DH subtypes to produce E double bonds
    """
    new_kr_type = 'B'
    new_dh_type = 'E'
    return new_kr_type, new_dh_type

def check_substrate_ez_compatibility(pks_features: dict, z_mod_num: List[int]) -> int:
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
    for mod in z_mod_num:
        substrate = pks_features['Substrate'][mod]
        old_kr_type = pks_features['KR Type'][mod]
        old_dh_type = pks_features['DH Type'][mod]
        if old_kr_type is not None and old_dh_type is not None:
            if substrate == 'Malonyl-CoA':
                new_kr_type, new_dh_type = z_kr_dh_combo()
                if new_kr_type != old_kr_type:
                    pks_features['KR Type'][mod] = new_kr_type
                    pks_features['DH Type'][mod] = new_dh_type
                    print(f"    Making KR-DH swap in module {mod}")
                    print(f"    M{mod}: Swapped KR-DH types from {old_kr_type}-{old_dh_type} to {new_kr_type}-{new_dh_type}")
    for mod in e_mod_num:
        substrate = pks_features['Substrate'][mod]
        old_kr_type = pks_features['KR Type'][mod]
        old_dh_type = pks_features['DH Type'][mod]
        if old_kr_type is not None and old_dh_type is not None:
            if substrate == 'Malonyl-CoA':
                new_kr_type, new_dh_type = e_kr_dh_combo()
                if new_kr_type != old_kr_type:
                    pks_features['KR Type'][mod] = new_kr_type
                    pks_features['DH Type'][mod] = new_dh_type
                    print(f"    Making KR-DH swap in module {mod}")
                    print(f"    M{mod}: Swapped KR-DH types from {old_kr_type}-{old_dh_type} to {new_kr_type}-{new_dh_type}")
    return pks_features

def apply_dh_swaps(target_mol: Chem.Mol, pks_features: dict, full_mapping_df: pd.DataFrame) -> dict:
    """
    Make KR-DH subtype swaps to correct alkene stereochemistry

    Args:
        target_mol (Chem.Mol): User-defined target molecule
        pks_features (dict): PKS design features (domains, domain subtypes, substrates, etc.)
        full_mapping_df (pd.DataFrame): Contains mapping of atoms between PKS product and target molecule
    
    Returns:
        dh_swapped_pks_features (dict): PKS features with updated KR-DH subtypes
    """
    z_bonds, e_bonds = extract_ez_atoms(target_mol)
    z_mods, e_mods = identify_dh_modules(full_mapping_df, z_bonds, e_bonds)
    check_substrate_ez_compatibility(pks_features, z_mods)
    dh_swapped_pks_features = correct_ez_stereo(pks_features, z_mods, e_mods)
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
