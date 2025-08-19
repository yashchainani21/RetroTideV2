import bcs
from retrotide import structureDB
from typing import Optional, List
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, MolStandardize, rdFMCS, rdmolops
from itertools import product
from stereopostprocessing import fingerprints, similarity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_bcs_info(pks_design)-> dict:
    pks_features = {
        'Module Number': [],
        'Substrate': [],
        'KR Type': [],
        'DH' : [],
        'ER' : []
    }

    for idx, module in enumerate(pks_design):
        pks_features['Module Number'].append(idx)
        
        substrate = module.domains[bcs.AT].substrate
        pks_features['Substrate'].append(substrate)

        if bcs.KR in module.domains:
            type = module.domains[bcs.KR].type
            pks_features['KR Type'].append(type)

            if bcs.DH in module.domains:
                pks_features['DH'].append(True)
            else:
                pks_features['DH'].append(False)
        
            if bcs.ER in module.domains:
                pks_features['ER'].append(True)
            else:
                pks_features['ER'].append(False)
        else:
            pks_features['KR Type'].append(None)
            pks_features['DH'].append(False)
            pks_features['ER'].append(False)

    return pks_features

def find_adjacent_backbone_carbon_pairs(mol: Chem.Mol, fully_mapped_df: pd.DataFrame) -> list:
    # Create a mapping from atom index to module
    atom_to_module = {}
    for _, row in fully_mapped_df.iterrows():
        atom_idx = int(row['Product Atom Idx'])
        module = row['Module']
        atom_to_module[atom_idx] = module
    
    # Find all carbon atoms
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
                    pair = tuple(sorted([
                        (carbon_idx, carbon_module),
                        (neighbor_idx, neighbor_module)
                    ], key=lambda x: x[0]))
                    
                    if pair not in adjacent_pairs:
                        adjacent_pairs.append(pair)
    
    return adjacent_pairs

def get_module_order(module_name: str) -> int:
    if module_name == 'LM':
        return 0
    elif module_name.startswith('M'):
        return int(module_name[1:])
    else:
        return float('inf')  # Unknown modules go to end

def filter_sequential_module_pairs(adjacent_pairs: list) -> list:
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
    mismatch_pairs = []
    
    for atom1_idx, module1, atom2_idx, module2 in sequential_pairs:
        # Check if either atom is in the mismatch list
        if atom1_idx in mmatch1 or atom2_idx in mmatch1:
            mismatch_pairs.append((atom1_idx, module1, atom2_idx, module2))
    
    return mismatch_pairs

def check_alpha_carbon_and_hydroxyl_patterns(mol: Chem.Mol, mismatch_pairs: list) -> list:
    # Define SMARTS patterns
    alpha_carbon_patterns = {
        'alpha_to_carbonyl': '[C:1][C:2](=[O:3])',
        'alpha_to_hydroxyl': '[C:1][C:2][OH:3]',
        'alpha_to_ester': '[C:1][C:2](=[O:3])[O:4][C:5]'  # Added ester pattern
    }
    
    hydroxyl_carbon_pattern = '[C:1][OH:2]'
    
    # Pattern to detect ester-forming oxygen and its carbon substituent
    ester_oxygen_pattern = '[C:1](=[O:2])[O:3][C:4]'  # C=O-O-C ester pattern
    
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
        alpha_atoms = set()  # Track atoms that are alpha to something
        
        for pattern_name, pattern_smarts in alpha_carbon_patterns.items():
            pattern_mol = Chem.MolFromSmarts(pattern_smarts)
            matches = mol.GetSubstructMatches(pattern_mol)
            
            for match in matches:
                alpha_carbon_idx = match[0]
                if alpha_carbon_idx == atom1_idx:
                    alpha_atoms.add(atom1_idx)
                elif alpha_carbon_idx == atom2_idx:
                    alpha_atoms.add(atom2_idx)
        
        # Label alpha atoms as 'alpha_substituted'
        if atom1_idx in alpha_atoms:
            pair_info['atom1_patterns'].append('alpha_substituted')
        if atom2_idx in alpha_atoms:
            pair_info['atom2_patterns'].append('alpha_substituted')
        
        # Check hydroxyl substituted carbon pattern
        hydroxyl_pattern_mol = Chem.MolFromSmarts(hydroxyl_carbon_pattern)
        hydroxyl_matches = mol.GetSubstructMatches(hydroxyl_pattern_mol)
        
        for match in hydroxyl_matches:
            hydroxyl_carbon_idx = match[0]
            if hydroxyl_carbon_idx == atom1_idx:
                pair_info['atom1_patterns'].append('hydroxyl_substituted')
            elif hydroxyl_carbon_idx == atom2_idx:
                pair_info['atom2_patterns'].append('hydroxyl_substituted')
        
        # Check ester oxygen pattern - look for carbons bonded to ester-forming oxygen
        ester_pattern_mol = Chem.MolFromSmarts(ester_oxygen_pattern)
        ester_matches = mol.GetSubstructMatches(ester_pattern_mol)
        
        for match in ester_matches:
            # match[0] = carbonyl carbon, match[1] = carbonyl oxygen, 
            # match[2] = ester oxygen, match[3] = carbon bonded to ester oxygen
            ester_carbon_idx = match[3]  # The carbon substituent on the ester oxygen
            
            if ester_carbon_idx == atom1_idx:
                pair_info['atom1_patterns'].append('ester_substituted')
            elif ester_carbon_idx == atom2_idx:
                pair_info['atom2_patterns'].append('ester_substituted')
        
        results.append(pair_info)
    
    return results

def kr_swaps(pks_features: dict, pattern_results: list, mmatch1: list):
    """
    Update KR types based on chiral mismatches and their structural patterns.
    """
    
    for result in pattern_results:
        atom1_idx = result['atom1_idx']
        module1 = result['module1']
        atom2_idx = result['atom2_idx']
        module2 = result['module2']
        
        # Process both atoms if they are mismatched
        atoms_to_process = []
        
        if atom1_idx in mmatch1:
            atoms_to_process.append({
                'atom': atom1_idx,
                'module': module1,
                'patterns': result['atom1_patterns'],
                'adjacent_patterns': result['atom2_patterns']
            })
        
        if atom2_idx in mmatch1:
            atoms_to_process.append({
                'atom': atom2_idx,
                'module': module2,
                'patterns': result['atom2_patterns'],
                'adjacent_patterns': result['atom1_patterns']
            })
        
        # Process each mismatched atom
        for atom_info in atoms_to_process:
            mismatched_atom = atom_info['atom']
            mismatched_module = atom_info['module']
            mismatched_patterns = atom_info['patterns']
            adjacent_patterns = atom_info['adjacent_patterns']

            # Check patterns for the mismatched atom
            is_alpha = 'alpha_substituted' in mismatched_patterns
            is_hydroxyl = 'hydroxyl_substituted' in mismatched_patterns
            is_ester = 'ester_substituted' in mismatched_patterns
            
            # Determine target module based on pattern
            if is_hydroxyl:
                if mismatched_module == 'LM':
                    target_module_number = 1  # LM + 1 = Module 1
                elif mismatched_module.startswith('M'):
                    carbon_module_number = int(mismatched_module[1:])
                    target_module_number = carbon_module_number + 1
                else:
                    print(f"Unknown module format: {mismatched_module}")
                    continue
 
            else:
                if mismatched_module == 'LM':
                    target_module_number = 1
                elif mismatched_module.startswith('M'):
                    target_module_number = int(mismatched_module[1:])
                else:
                    print(f"Unknown module format: {mismatched_module}")
                    continue

            if target_module_number >= len(pks_features['Module Number']):
                print(f"Target module {target_module_number} not found in PKS design")
                continue
                
            old_kr_type = pks_features['KR Type'][target_module_number]
            
            print(f"------Analyzing mismatch in module {mismatched_module}-------")
            print(f"  Mismatched atom {mismatched_atom}: alpha={is_alpha}, hydroxyl={is_hydroxyl}")
            print(f"  Will modify KR type in module {target_module_number}")
            
            # Simple case: Malonyl-CoA with no DH
            if (pks_features['Substrate'][target_module_number] == "Malonyl-CoA" and 
                pks_features['DH'][target_module_number] == False):
                if old_kr_type == 'A':
                    new_kr_type = 'B'
                elif old_kr_type == 'B':
                    new_kr_type = 'A'
                else:
                    print(f"Module {target_module_number} has unexpected KR type: {old_kr_type}")
                    continue

            elif (pks_features['Substrate'][target_module_number] == "Malonyl-CoA" and 
                pks_features['DH'][target_module_number] == True):
                new_kr_type = 'B'

            elif (pks_features['Substrate'][target_module_number] == "Methylmalonyl-CoA" and 
                pks_features['DH'][target_module_number] == True):
                new_kr_type = 'B1'
            
            # Complex case: Use pattern analysis
            else:
                if old_kr_type is None:
                    if is_alpha:
                        new_kr_type = 'C1'
                        print(f"  Case 1: Adding KR type C1 for alpha carbon mismatch")
                    else:
                        print(f"  No KR domain and not alpha carbon - cannot fix")
                        continue
                
                else:
                    # Parse existing KR type (e.g., 'A1' -> letter='A', number='1')
                    if len(old_kr_type) >= 2:
                        letter = old_kr_type[0]
                        number = old_kr_type[1:]
                    else:
                        letter = old_kr_type
                        number = '1'  # Default number
                        
                    # Determine new KR type based on patterns
                    if is_alpha and is_hydroxyl:
                        # Case 4: Both alpha and -OH are wrong, swap both letter and number
                        new_letter = 'B' if letter == 'A' else 'A'
                        new_number = '2' if number == '1' else '1'
                        new_kr_type = new_letter + new_number
                        print(f"  Case 4: Both patterns wrong - swapping both letter and number")
                    elif is_alpha and is_ester:
                        # Case 6: Alpha is wrong and -O- is wrong, swap both letter and number
                        new_letter = 'B' if letter == 'A' else 'A'
                        new_number = '2' if number == '1' else '1'
                        new_kr_type = new_letter + new_number
                        print(f"  Case 6: Alpha wrong, O- wrong - swapping both letter and number")
                    elif is_alpha and not is_hydroxyl:
                        # Case 2: Alpha is wrong and -OH is right, swap the # and keep the letter
                        new_number = '2' if number == '1' else '1'
                        new_kr_type = letter + new_number
                        print(f"  Case 2: Alpha wrong, hydroxyl right - swapping number only")
                    elif is_alpha and not is_ester:
                        # Case 5: Alpha is wrong and -O- is right, swap the # and keep the letter
                        new_number = '2' if number == '1' else '1'
                        new_kr_type = letter + new_number
                        print(f"  Case 5: Alpha wrong, O- right - swapping # only")
                    elif not is_alpha and is_hydroxyl:
                        # Case 3: Alpha is right and -OH is wrong, swap the letter and keep the #
                        new_letter = 'B' if letter == 'A' else 'A'
                        new_kr_type = new_letter + number
                        print(f"  Case 3: Alpha right, hydroxyl wrong - swapping letter only")
                    elif not is_alpha and is_ester:
                        # Case 7: Alpha is right and -O- is wrong, swap the letter and keep the #
                        new_letter = 'B' if letter == 'A' else 'A'
                        new_kr_type = new_letter + number
                        print(f"  Case 7: Alpha right, O- wrong - swapping letter only")
                    else:
                        # Neither pattern matches - use simple A/B swap
                        new_kr_type = 'B' if letter == 'A' else 'A'
                        print(f"  Default case: Simple A/B swap")
                
            # Update the KR type in the target module
            pks_features['KR Type'][target_module_number] = new_kr_type
            print(f"------Updating Module {target_module_number}------")
            print(f"Updated KR type from {old_kr_type} to {new_kr_type}")
            print(f"     ")
        
    return pks_features

def new_design(pks_features:dict) -> Chem.Mol:
    modules = []
    for idx, substrate in enumerate(pks_features['Substrate']):
        substrate = pks_features['Substrate'][idx]
        if idx == 0:
            domains_dict = OrderedDict({bcs.AT: bcs.AT(active = True, substrate = substrate)})
            module = bcs.Module(domains = domains_dict, loading = True)
            modules.append(module)
        else:
            AT = bcs.AT(active = True, substrate = substrate)
            domains_dict = OrderedDict({bcs.AT: AT})
            if pks_features['KR Type'][idx] is not None:
                domains_dict.update({bcs.KR: bcs.KR(active = True, type = pks_features['KR Type'][idx])})
            if pks_features['DH'][idx]:
                domains_dict.update({bcs.DH: bcs.DH(active = True)})
            if pks_features['ER'][idx]:
                domains_dict.update({bcs.ER: bcs.ER(active = True)})

            module = bcs.Module(domains = domains_dict, loading = False)
            modules.append(module)
    cluster_f = bcs.Cluster(modules = modules)
    pks_product = cluster_f.computeProduct(structureDB)
    return pks_product, cluster_f

def map_and_offload_corrected_product(cluster_f: bcs.Cluster, pks_product: Chem.Mol, target_mol: Chem.Mol, offload_mechanism: str):
    

    # Map the product to the target molecule
    mapped_product = mcs_map_product_to_target(pks_product, target_mol)

    # Offload the mapped product
    offloaded_product = offload_pks_product(mapped_product, target_mol, 'thiolysis')

    return offloaded_product

def check_similarity(final_corrected_prod: Chem.Mol, target_mol: Chem.Mol) -> float:
    fp_prod = fingerprints.get_fingerprint(Chem.MolToSmiles(final_corrected_prod), 'mapchiral')
    fp_targ = fingerprints.get_fingerprint(Chem.MolToSmiles(target_mol), 'mapchiral')

    score = similarity.get_similarity(fp_prod, fp_targ, 'jaccard')
    return score

