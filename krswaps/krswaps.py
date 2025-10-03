"""
Module to handle KR swaps based on chiral mismatches between the PKS product and target molecule
"""
# pylint: disable=no-member
from collections import OrderedDict
from dataclasses import dataclass
from rdkit import Chem
import pandas as pd
import bcs
from retrotide import structureDB

@dataclass
class ChiralFlags:
    """
    Data class to hold flags for chiral center mismatches
    for each pair
    """
    alpha_cc: bool = False
    beta_cc: bool = False

def get_bcs_info(pks_design: list)-> dict:
    """
    Constructs a dictionary containing information about the PKS design

    Args:
        pks_design (list): Initial RetroTide design

    Returns:
        pks_features (dict): Dictionary with information about the intial PKS design,
        including module number, substrate, KR, DH, and ER subtypes, and whether DH
        or ER domains are present.
    """
    pks_features = {
        'Module Number': [],
        'Substrate': [],
        'KR Type': [],
        'DH Type' : [],
        'ER Type' : [],
        'KR' : [],
        'DH' : [],
        'ER' : []
    }
    for idx, module in enumerate(pks_design):
        pks_features['Module Number'].append(idx)
        substrate = module.domains[bcs.AT].substrate
        pks_features['Substrate'].append(substrate)
        if bcs.KR in module.domains:
            kr_type = module.domains[bcs.KR].type
            pks_features['KR Type'].append(kr_type)
            pks_features['KR'].append(True)
            if bcs.DH in module.domains:
                dh_type = module.domains[bcs.DH].type
                pks_features['DH Type'].append(dh_type)
                pks_features['DH'].append(True)
            else:
                pks_features['DH'].append(False)
                pks_features['DH Type'].append(None)        
            if bcs.ER in module.domains:
                er_type = module.domains[bcs.ER].type
                pks_features['ER Type'].append(er_type)
                pks_features['ER'].append(True)
            else:
                pks_features['ER'].append(False)
                pks_features['ER Type'].append(None)
        else:
            pks_features['KR Type'].append(None)
            pks_features['DH Type'].append(None)
            pks_features['ER Type'].append(None)
            pks_features['DH'].append(False)
            pks_features['ER'].append(False)
            pks_features['KR'].append(False)
    return pks_features

def extract_pairs(mol: Chem.Mol, fully_mapped_df: pd.DataFrame) -> list:
    """
    Finds pairs of backbone carbon atoms mapped to different modules
    in the PKS product
    """
    # Track atom-module mapping
    atom_to_module = {}
    for _, row in fully_mapped_df.iterrows():
        atom_idx = int(row['Product Atom Idx'])
        module = row['Module']
        atom_to_module[atom_idx] = module
    c_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    # Find pairs of backbone carbons from different modules
    backbone_pairs = []
    for c_idx in c_atoms:
        if c_idx not in atom_to_module:
            continue
        carbon_atom = mol.GetAtomWithIdx(c_idx)
        carbon_module = atom_to_module[c_idx]
        # Assess which carbons make up the backbone
        for neighbor in carbon_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor.GetSymbol() == 'C' and neighbor_idx in atom_to_module:
                neighbor_module = atom_to_module[neighbor_idx]
                if carbon_module != neighbor_module:
                    # Create ordered pair
                    pair = tuple(sorted([(c_idx, carbon_module),
                                        (neighbor_idx, neighbor_module)],
                                        key=lambda x: x[0]))                    
                    if pair not in backbone_pairs:
                        backbone_pairs.append(pair)
    return backbone_pairs

def get_module_number(module_name: str) -> int:
    """
    Helper function to extract module number

    Returns:
        module_number (int): 'LM' as 0 and 'M1' as 1, etc.
    """
    if module_name == 'LM':
        module_number = 0
    else:
        module_number = module_name.lstrip('M')
    return int(module_number)

def filter_pairs(backbone_pairs: list) -> list:
    """
    Uses the module order to extract pairs of adjacent carbon atoms that are from
    sequential PKS modules
    """
    sequential_pairs = []    
    for (atom1_idx, module1), (atom2_idx, module2) in backbone_pairs:
        modi = get_module_number(module1)
        modj = get_module_number(module2)
        # Check if modules are sequential and order like (Mi, Mi+1)
        if abs(modi - modj) == 1:
            if modi < modj:
                sequential_pairs.append((atom1_idx, module1, atom2_idx, module2))
            else:
                sequential_pairs.append((atom2_idx, module2, atom1_idx, module1))
    sequential_pairs.sort(key=lambda x: get_module_number(x[1]))
    return sequential_pairs

def identify_pairs_with_mismatches(sequential_pairs: list, mmatch1: list) -> list:
    """
    Cross references each pair with the list of mismatched chiral centers

    Args:
        sequential_pairs (list): List of tuples with pairs of backbone carbon atoms
        mmatch1 (list): List of atom indices with chiral mismatches

    Returns:
        mismatch_pairs (list): List of pairs that have at least one chiral mismatch
    """
    mismatch_pairs = []
    for atom1_idx, module1, atom2_idx, module2 in sequential_pairs:
        if atom1_idx in mmatch1 or atom2_idx in mmatch1:
            mismatch_pairs.append((atom1_idx, module1, atom2_idx, module2))
    return mismatch_pairs

def check_alpha_carbon(mol: Chem.Mol, atom1_idx: int, atom2_idx: int) -> dict:
    """
    Check if a carbon is alpha to a carbonyl, hydroxyl, ether,
    or ester group
    """
    alpha_carbon_patterns = [
        '[C:1][C:2](=[O:3])',
        '[C:1][C:2][OH:3]',
        '[C:1][C:2][O:3]',
        '[C:1][C:2](=[O:3])[O:4][C:5]',
        '[C:1]/[C:2]=[C:3]/[C:4]',
        '[C:1]/[C:2]=[C:3]\[C:4]',
        '[C:1]([CH2:2])[CH2:3]'
    ]
    results = {'atom1' : [], 'atom2': []}
    for pattern in alpha_carbon_patterns:
        pattern_mol = Chem.MolFromSmarts(pattern)
        matches = mol.GetSubstructMatches(pattern_mol)
        for match in matches:
            alpha_carbon_idx = match[0]
            if alpha_carbon_idx == atom1_idx:
                results['atom1'].append('alpha_substituted')
            elif alpha_carbon_idx == atom2_idx:
                results['atom2'].append('alpha_substituted')
    return results

def check_hydroxyl_substituted(mol: Chem.Mol, atom1_idx: int, atom2_idx: int) -> dict:
    """
    Check if a carbon is hydroxyl substituted
    """
    hydroxyl_carbon_pattern = '[C:1][OH:2]'
    results = {'atom1' : [], 'atom2': []}
    pattern_mol = Chem.MolFromSmarts(hydroxyl_carbon_pattern)
    matches = mol.GetSubstructMatches(pattern_mol)
    for match in matches:
        hydroxyl_carbon_idx = match[0]
        if hydroxyl_carbon_idx == atom1_idx:
            results['atom1'].append('hydroxyl_substituted')
        elif hydroxyl_carbon_idx == atom2_idx:
            results['atom2'].append('hydroxyl_substituted')
    return results

def check_ester_substituted(mol: Chem.Mol, atom1_idx: int, atom2_idx: int) -> dict:
    """
    Check if a carbon is ester substituted
    """
    ester_oxygen_pattern = '[C:1](=[O:2])[O:3][C:4]'
    results = {'atom1' : [], 'atom2': []}
    pattern_mol = Chem.MolFromSmarts(ester_oxygen_pattern)
    matches = mol.GetSubstructMatches(pattern_mol)
    for match in matches:
        ester_carbon_idx = match[3]
        if ester_carbon_idx == atom1_idx:
            results['atom1'].append('ester_substituted')
        elif ester_carbon_idx == atom2_idx:
            results['atom2'].append('ester_substituted')
    return results

def check_substituent_patterns(mol: Chem.Mol, mismatch_pairs: list) -> list:
    """
    Check substituent patterns for each pair including one
    or more mismatched chiral centers
    """
    pattern_results = []
    for atom1_idx, module1, atom2_idx, module2 in mismatch_pairs:
        pair_info = {
            'atom1_idx': atom1_idx,
            'module1': module1,
            'atom2_idx': atom2_idx,
            'module2': module2,
            'atom1_patterns': [],
            'atom2_patterns': []
    }
        # Check alpha carbon patterns
        alpha_results = check_alpha_carbon(mol, atom1_idx, atom2_idx)
        pair_info['atom1_patterns'].extend(alpha_results['atom1'])
        pair_info['atom2_patterns'].extend(alpha_results['atom2'])
        
        # Check hydroxyl substituted carbon pattern
        hydroxyl_results = check_hydroxyl_substituted(mol, atom1_idx, atom2_idx)
        pair_info['atom1_patterns'].extend(hydroxyl_results['atom1'])
        pair_info['atom2_patterns'].extend(hydroxyl_results['atom2'])
        
        # Check ester pattern
        ester_results = check_ester_substituted(mol, atom1_idx, atom2_idx)
        pair_info['atom1_patterns'].extend(ester_results['atom1'])
        pair_info['atom2_patterns'].extend(ester_results['atom2'])

        pattern_results.append(pair_info)
    return pattern_results

def kr_type_logic(pks_features: dict, target_module_number: int,
                  old_kr_type: str, flags: ChiralFlags) -> str:
    """
    Apply logic for identifying the correct KR type
    """
    substrate = pks_features['Substrate'][target_module_number]
    new_kr_type = None
    # Simple cases
    if (substrate == "Malonyl-CoA" and 
        pks_features['ER'][target_module_number] is False):
        if old_kr_type == 'A':
            new_kr_type = 'B'
        else:
            new_kr_type = 'A'
    elif (substrate == "Malonyl-CoA" and 
        pks_features['ER'][target_module_number] is True):
        new_kr_type = 'B'
    elif (substrate != "Malonyl-CoA" and 
        pks_features['DH'][target_module_number] is True):
        new_kr_type = 'B1'
    # Complex cases
    elif old_kr_type is None:
        if flags.alpha_cc:
            new_kr_type = 'C2'
            print("    Case 1: Adding KR type C2 to perform epimerization of alpha chiral center")
        else:
            print("    No KR domain and not alpha carbon - cannot fix by KR swap")
    else:
        # Parse existing KR type (e.g., 'A1' -> letter='A', number='1')
        letter = old_kr_type[0]
        number = old_kr_type[1:] if len(old_kr_type) >= 2 else '1'
        if flags.alpha_cc and flags.beta_cc:
            new_letter = 'B' if letter == 'A' else 'A'
            new_number = '2' if number == '1' else '1'
            new_kr_type = new_letter + new_number
            print("    Case 2: Both wrong - swapping both letter and number")
        elif flags.alpha_cc and not flags.beta_cc:
            new_number = '2' if number == '1' else '1'
            new_kr_type = letter + new_number
            print("    Case 3: Alpha wrong, Beta right - swapping number only")
        elif not flags.alpha_cc and flags.beta_cc:
            new_letter = 'B' if letter == 'A' else 'A'
            new_kr_type = new_letter + number
            print("    Case 4: Alpha right, Beta wrong - swapping letter only")
        else:
            new_kr_type = old_kr_type
            print("    No patterns matched - returning original KR type")
    return new_kr_type

def identify_new_kr_type(pair_info: dict, mmatch1: list, pks_features: dict):
    """
    Assess if the mismatched chiral center is alpha or beta
    Select the correct KR swap to make accordingly
    """
    atom1_idx = pair_info['atom1_idx']
    atom2_idx = pair_info['atom2_idx']
    module1 = get_module_number(pair_info['module1'])
    module2 = get_module_number(pair_info['module2'])
    patterns1 = pair_info['atom1_patterns']
    patterns2 = pair_info['atom2_patterns']

    if atom1_idx in mmatch1 and atom2_idx in mmatch1:
        alpha_cc = True
        beta_cc = True
    elif atom1_idx in mmatch1:
        alpha_cc = 'alpha_substituted' in patterns1
        beta_cc = ('hydroxyl_substituted' in patterns1) or \
                   ('ester_substituted' in patterns1)
    elif atom2_idx in mmatch1:
        alpha_cc = 'alpha_substituted' in patterns2
        beta_cc = ('hydroxyl_substituted' in patterns2) or \
                   ('ester_substituted' in patterns2)
    else:
        raise ValueError("Mismatched pairs were not assessed correctly")

    target_module_number = module1 if module1 > module2 else module2
    old_kr_type = pks_features['KR Type'][target_module_number]
    print(f"----Analyzing mismatch(s) in pair ({atom1_idx}, {atom2_idx})-----")
    print(f"    alpha={alpha_cc}, beta={beta_cc}")
    new_kr_type = kr_type_logic(pks_features, target_module_number,
                                         old_kr_type,
                                         flags=ChiralFlags(alpha_cc=alpha_cc,
                                                           beta_cc=beta_cc
                                                           ))
    if new_kr_type != old_kr_type:
        pks_features['KR Type'][target_module_number] = new_kr_type
        print(f"    Making KR swap in module {target_module_number}")
        print(f"    M{target_module_number}: Swapped KR type from {old_kr_type} to {new_kr_type}")

def kr_swaps(pks_features: dict, mmatch1: list, pattern_results: list) -> dict:
    """
    Update KR types based on chiral mismatches and their structural patterns
    """
    for pair_info in pattern_results:
        identify_new_kr_type(pair_info, mmatch1, pks_features)
    return pks_features

def new_design(pks_features:dict) -> Chem.Mol:
    """
    Reconstruct PKS design based on updated KR types

    Returns:
        pks_product (Chem.Mol): The final PKS product after stereocorrections
    """
    modules = []
    for idx, substrate in enumerate(pks_features['Substrate']):
        substrate = pks_features['Substrate'][idx]
        if idx == 0:
            domains_dict = OrderedDict({bcs.AT: bcs.AT(active=True, substrate=substrate)})
            module = bcs.Module(domains=domains_dict, loading=True)
            modules.append(module)
        else:
            at = bcs.AT(active=True, substrate=substrate)
            domains_dict = OrderedDict({bcs.AT: at})
            if pks_features['KR Type'][idx] is not None:
                domains_dict.update({bcs.KR: bcs.KR(active=True,
                                                    type=pks_features['KR Type'][idx])})
            if pks_features['DH Type'][idx] is not None:
                domains_dict.update({bcs.DH: bcs.DH(active=True,
                                                    type=pks_features['DH Type'][idx])})
            if pks_features['ER Type'][idx] is not None:
                domains_dict.update({bcs.ER: bcs.ER(active=True, type=pks_features['ER Type'][idx])})
            module = bcs.Module(domains=domains_dict, loading=False)
            modules.append(module)
    cluster_f = bcs.Cluster(modules=modules)
    pks_product = cluster_f.computeProduct(structureDB)
    return pks_product, modules

def apply_kr_swaps(unbound_product: Chem.Mol, full_mapping_df: pd.DataFrame,
                   mmatch1: list, pks_features: dict) -> dict:
    """
    Parse pairs of backbone carbons mapped to sequential modules and
    apply KR swaps
    """
    pairs = extract_pairs(unbound_product, full_mapping_df)
    sequential_pairs = filter_pairs(pairs)
    pairs_with_mismatches = identify_pairs_with_mismatches(sequential_pairs, mmatch1)
    pattern_results = check_substituent_patterns(unbound_product, pairs_with_mismatches)
    pks_features_updated = kr_swaps(pks_features, mmatch1, pattern_results)
    return pks_features_updated

def apply_er_swaps(unbound_product: Chem.Mol, full_mapping_df: pd.DataFrame,
                   mmatch1_f: list, pks_features: dict) -> dict:
    """
    If remaining alpha substituted chiral mismatches and an ER domain is present,
    swap the ER type
    """
    pairs = extract_pairs(unbound_product, full_mapping_df)
    sequential_pairs = filter_pairs(pairs)
    pairs_with_mismatches = identify_pairs_with_mismatches(sequential_pairs, mmatch1_f)
    pattern_results = check_substituent_patterns(unbound_product, pairs_with_mismatches)
    for pair_info in pattern_results:
        atom1_idx = pair_info['atom1_idx']
        atom2_idx = pair_info['atom2_idx']
        module1 = get_module_number(pair_info['module1'])
        module2 = get_module_number(pair_info['module2'])
        patterns1 = pair_info['atom1_patterns']
        patterns2 = pair_info['atom2_patterns']
        alpha_cc = 'alpha_substituted' in patterns1 or 'alpha_substituted' in patterns2

        target_module_number = module1 if module1 > module2 else module2
        if alpha_cc:
            print(f"----Analyzing mismatch in pair ({atom1_idx}, {atom2_idx})-----")
            print(f"    alpha={alpha_cc}")
            if pks_features['ER'][target_module_number]:
                old_er_type = pks_features['ER Type'][target_module_number]
                new_er_type = 'D' if old_er_type == 'L' else 'L'
                pks_features['ER Type'][target_module_number] = new_er_type
                print(f"    Making ER swap in module {target_module_number}")
                print(f"    M{target_module_number}: Swapped ER type from {old_er_type} to {new_er_type}")
            else:
                print(f"    No ER domain in module {target_module_number} - cannot swap ER type")
    return pks_features

def check_swaps_accuracy(match: list, mmatch: list) -> float:
    """
    Assess stereochemistry correspondence post implementing KR Swaps
    and ER Swaps
    """
    if (len(match) + len(mmatch)) > 0:
        swaps_score = len(match)/(len(match)+len(mmatch))
    else:
        swaps_score = None
    return swaps_score
