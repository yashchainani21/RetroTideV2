'''
Module to handle KR swaps based on chiral mismatches between the PKS product and target molecule

@author: Kenna Roberts
'''
# pylint: disable=no-member
from collections import OrderedDict
from dataclasses import dataclass
from rdkit import Chem
import pandas as pd
import bcs
from retrotide import structureDB

@dataclass
class PatternFlags:
    """
    Data class to hold flags for different substituent patterns
    """
    is_alpha: bool = False
    is_hydroxyl: bool = False
    is_ester: bool = False

def get_bcs_info(pks_design: list)-> dict:
    """
    Constructs a dictionary containing information about the PKS design

    Args:
        pks_design (list): Initial RetroTide design

    Returns:
        pks_features (dict): Dictionary with information about the intial PKS design,
        including module number, substrate, KR type, and whether DH or ER domains are present.
    """
    pks_features = {
        'Module Number': [],
        'Substrate': [],
        'KR Type': [],
        'DH Type' : [],
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
            if bcs.DH in module.domains:
                dh_type = module.domains[bcs.DH].type
                pks_features['DH Type'].append(dh_type)
                pks_features['DH'].append(True)
            else:
                pks_features['DH'].append(False)
                pks_features['DH Type'].append(None)        
            if bcs.ER in module.domains:
                pks_features['ER'].append(True)
            else:
                pks_features['ER'].append(False)
        else:
            pks_features['KR Type'].append(None)
            pks_features['DH Type'].append(None)
            pks_features['DH'].append(False)
            pks_features['ER'].append(False)
    return pks_features

def find_adjacent_backbone_carbon_pairs(mol: Chem.Mol, fully_mapped_df: pd.DataFrame) -> list:
    """
    Finds pairs of adjacent backbone carbon atoms in the PKS product
    """
    # Create a mapping from atom index to module
    atom_to_module = {}
    for _, row in fully_mapped_df.iterrows():
        atom_idx = int(row['Product Atom Idx'])
        module = row['Module']
        atom_to_module[atom_idx] = module
    carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    # Find adjacent carbon pairs from different modules
    adjacent_pairs = []
    for carbon_idx in carbon_atoms:
        if carbon_idx not in atom_to_module:
            continue
        carbon_atom = mol.GetAtomWithIdx(carbon_idx)
        carbon_module = atom_to_module[carbon_idx]
        # Check all neighboring atoms
        for neighbor in carbon_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            # Only consider carbon neighbors
            if neighbor.GetSymbol() == 'C' and neighbor_idx in atom_to_module:
                neighbor_module = atom_to_module[neighbor_idx]
                # Only include if from different modules
                if carbon_module != neighbor_module:
                    # Create ordered pair to avoid duplicates (smaller index first)
                    pair = tuple(sorted([(carbon_idx, carbon_module),
                                        (neighbor_idx, neighbor_module)],
                                        key=lambda x: x[0]))                    
                    if pair not in adjacent_pairs:
                        adjacent_pairs.append(pair)
    return adjacent_pairs

def get_module_order(module_name: str) -> int:
    """
    Helper function to extract module order

    Returns:
        module_number (int): 'LM' as 0 and 'M1' as 1, etc.
    """
    if module_name == 'LM':
        module_number = 0
    elif module_name.startswith('M'):
        module_number = int(module_name[1:])
    else:
        raise ValueError(f"Unknown module format: {module_name}")
    return module_number

def filter_sequential_module_pairs(adjacent_pairs: list) -> list:
    """
    Uses the module order to extract pairs of adjacent carbon atoms that are from
    sequential PKS modules
    """
    sequential_pairs = []    
    for (atom1_idx, module1), (atom2_idx, module2) in adjacent_pairs:
        order1 = get_module_order(module1)
        order2 = get_module_order(module2)
        
        # Check if modules are sequential (difference of 1)
        if abs(order1 - order2) == 1:
            # Order so that earlier module comes first
            if order1 < order2:
                sequential_pairs.append((atom1_idx, module1, atom2_idx, module2))
            else:
                sequential_pairs.append((atom2_idx, module2, atom1_idx, module1))
    # Sort by the first module's order
    sequential_pairs.sort(key=lambda x: get_module_order(x[1]))
    return sequential_pairs

def report_pairs_with_chiral_mismatches(sequential_pairs: list, mmatch1: list) -> list:
    """
    Cross references each pairs with the list of mismatched chiral centers
    """
    mismatch_pairs = []
    for atom1_idx, module1, atom2_idx, module2 in sequential_pairs:
        # Check if either atom is in the mismatch list
        if atom1_idx in mmatch1 or atom2_idx in mmatch1:
            mismatch_pairs.append((atom1_idx, module1, atom2_idx, module2))
    return mismatch_pairs

def check_alpha_carbon(mol: Chem.Mol, atom1_idx: int, atom2_idx: int):
    """
    Check if a carbon is alpha to a carbonyl, hydroxyl, or ester group
    """
    alpha_carbon_patterns = [
        '[C:1][C:2](=[O:3])',
        '[C:1][C:2][OH:3]',
        '[C:1][C:2](=[O:3])[O:4][C:5]',
        '[C:1][C:2]=[C:3]' 
    ]
    results = {'atom1' : [], 'atom2': []}
    for smarts in alpha_carbon_patterns:
        pattern_mol = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern_mol)
            
        for match in matches:
            alpha_carbon_idx = match[0]
            if alpha_carbon_idx == atom1_idx:
                results['atom1'].append('alpha_substituted')
            elif alpha_carbon_idx == atom2_idx:
                results['atom2'].append('alpha_substituted')
    return results

def check_hydroxyl_substituted(mol: Chem.Mol, atom1_idx: int, atom2_idx: int):
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

def check_ester_substituted(mol: Chem.Mol, atom1_idx: int, atom2_idx: int):
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
    results = []
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

        results.append(pair_info)
    return results

def identify_module_to_edit(mismatched_module: str, is_hydroxyl: bool, is_ester: bool) -> int:
    """
    Identify which module to perform a KR swap on
    """
    if mismatched_module == 'LM':
        target_module =  1
    if mismatched_module.startswith('M'):
        target_module = int(mismatched_module[1:])
        if is_hydroxyl or is_ester:
            target_module += 1
    return target_module

def check_atoms_to_process(results: list, mmatch1: list) -> list:
    """
    Identify which atoms correspond to modules needing a KR swap
    """
    atoms_to_process = []
    for pair in results:
        atom1_idx = pair['atom1_idx']
        module1 = pair['module1']
        atom2_idx = pair['atom2_idx']
        module2 = pair['module2']
        
        if atom1_idx in mmatch1:
            atoms_to_process.append({
                'atom': atom1_idx,
                'module': module1,
                'patterns': pair['atom1_patterns'],
            })
        if atom2_idx in mmatch1:
            atoms_to_process.append({
                'atom': atom2_idx,
                'module': module2,
                'patterns': pair['atom2_patterns'],
            })
    return atoms_to_process

def determine_best_kr_type(pks_features: dict, target_module_number: int,
                           old_kr_type: str, flags: PatternFlags):
    """
    Apply logic for identifying the correct KR type
    """
    substrate = pks_features['Substrate'][target_module_number]
    new_kr_type = None
    # Simple case: Malonyl-CoA with no DH
    if (substrate == "Malonyl-CoA" and 
        pks_features['DH'][target_module_number] is False):
        if old_kr_type == 'A':
            new_kr_type = 'B'
        else:
            new_kr_type = 'A'
    elif (substrate == "Malonyl-CoA" and 
        pks_features['DH'][target_module_number] is True):
        if old_kr_type == 'A':
            new_kr_type = 'B'
        else:
            new_kr_type = 'A'
    elif (substrate != "Malonyl-CoA" and 
        pks_features['DH'][target_module_number] is True):
        new_kr_type = 'B1'
    # Complex cases
    elif old_kr_type is None:
        if flags.is_alpha:
            new_kr_type = 'C1'
            print("  Case 1: Adding KR type C1 to perform epimerization of alpha chiral center")
        else:
            print("  No KR domain and not alpha carbon - cannot fix by KR swap")
    else:
        # Parse existing KR type (e.g., 'A1' -> letter='A', number='1')
        letter = old_kr_type[0]
        number = old_kr_type[1:] if len(old_kr_type) >= 2 else '1'
        if flags.is_alpha and (flags.is_hydroxyl or flags.is_ester):
            new_letter = 'B' if letter == 'A' else 'A'
            new_number = '2' if number == '1' else '1'
            new_kr_type = new_letter + new_number
            print("  Case 2: Both patterns wrong - swapping both letter and number")
        elif flags.is_alpha and not (flags.is_hydroxyl or flags.is_ester):
            new_number = '2' if number == '1' else '1'
            new_kr_type = letter + new_number
            print("  Case 3: Alpha wrong, Beta right - swapping number only")
        elif not flags.is_alpha and (flags.is_hydroxyl or flags.is_ester):
            new_letter = 'B' if letter == 'A' else 'A'
            new_kr_type = new_letter + number
            print("  Case 4: Alpha right, Beta wrong - swapping letter only")
        else:
            new_kr_type = old_kr_type
            print("  No patterns matched - returning original KR type")
    return new_kr_type

def process_atom_info(atom_info: dict, pks_features: dict):
    """
    Assess if the mismatched chiral center is alpha or beta
    Select the correct KR swap to make accordingly
    """
    mismatched_atom = atom_info['atom']
    mismatched_module = atom_info['module']
    mismatched_patterns = atom_info['patterns']
    is_alpha = 'alpha_substituted' in mismatched_patterns
    is_hydroxyl = 'hydroxyl_substituted' in mismatched_patterns
    is_ester = 'ester_substituted' in mismatched_patterns
    target_module_number = identify_module_to_edit(mismatched_module, is_hydroxyl, is_ester)
    old_kr_type = pks_features['KR Type'][target_module_number]
    print(f"------Analyzing mismatch from {mismatched_module}-------")
    print(f"  Mismatched atom {mismatched_atom}: alpha={is_alpha}, beta={is_hydroxyl or is_ester}")
    print(f"  Will modify KR type in module {target_module_number}")
    new_kr_type = determine_best_kr_type(pks_features, target_module_number,
                                         old_kr_type,
                                         flags=PatternFlags(is_alpha=is_alpha,
                                                            is_hydroxyl=is_hydroxyl,
                                                            is_ester=is_ester))
    if new_kr_type != old_kr_type:
        pks_features['KR Type'][target_module_number] = new_kr_type
        print(f"------Updating M{target_module_number}------")
        print(f"Swapped KR type from {old_kr_type} to {new_kr_type}")

def kr_swaps(pks_features: dict, results: list, mmatch1: list) -> dict:
    """
    Update KR types based on chiral mismatches and their structural patterns
    """
    atoms = check_atoms_to_process(results, mmatch1)
    for atom_info in atoms:
        process_atom_info(atom_info, pks_features)
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
            if pks_features['ER'][idx]:
                domains_dict.update({bcs.ER: bcs.ER(active=True)})
            module = bcs.Module(domains=domains_dict, loading=False)
            modules.append(module)
    cluster_f = bcs.Cluster(modules=modules)
    pks_product = cluster_f.computeProduct(structureDB)
    return pks_product, modules

def apply_kr_swaps(unbound_product, full_mapping_df, mmatch1, pks_features):
    """
    Parse pairs of backbone carbons mapped to sequential modules and
    apply KR swaps
    """
    pairs = find_adjacent_backbone_carbon_pairs(unbound_product, full_mapping_df)
    sequential_pairs = filter_sequential_module_pairs(pairs)
    pairs_with_mismatches = report_pairs_with_chiral_mismatches(sequential_pairs, mmatch1)
    pattern_results = check_substituent_patterns(unbound_product, pairs_with_mismatches)
    pks_features_updated = kr_swaps(pks_features, pattern_results, mmatch1)
    return pks_features_updated

def check_swaps_accuracy(match, mmatch):
    """
    Assess stereochemistry correspondence post implementing KR Swaps
    """
    if (len(match) + len(mmatch)) > 0:
        swaps_score = len(match)/(len(match)+len(mmatch))
    else:
        swaps_score = None
    return swaps_score
