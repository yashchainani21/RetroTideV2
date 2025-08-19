import bcs
from typing import Optional, List
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFMCS
import pandas as pd
from stereopostprocessing import fingerprints, similarity

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    '''
    Modifies the starter and extender units available for RetroTide.
    Removes all starter and extender units not specifed in the input lists.
    '''
    for key in list(bcs.starters.keys()):
        if key not in starter_codes:
            bcs.starters.pop(key, None)
            
    for key in list(bcs.extenders.keys()):
        if key not in extender_codes:
            bcs.extenders.pop(key, None)
    return

modify_bcs_starters_extenders(starter_codes = ['trans-1,2-CPDA'],
                              extender_codes = ['Malonyl-CoA',
                                                'Methylmalonyl-CoA'])
from retrotide import structureDB, designPKS, compareToTarget
from stereopostprocessing import atommappks

def canonicalize_smiles(molecule_str: str, stereo: str) -> str:
    '''
    Standardizes a SMILES string and returns its canonical form with the
    specified isomeric information.
    
    Args:
        molecule_str (str): The SMILES string of the target molecule
        stereo (str): The type of stereochemistry to specify

    Returns:
        str: The canonicalized SMILES string.
    '''
    mol = Chem.MolFromSmiles(molecule_str)
    if stereo == 'R/S':
        target = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True).replace('/', '').replace('\\', '')
    elif stereo == 'E/Z':
        target = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True).replace('@', '').replace('@@', '')
    elif stereo == 'none':
        target = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    elif stereo == 'all':
        target = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) 
    else:
        raise ValueError("Invalid stereo option. Choose from 'R/S', 'E/Z', 'none', or 'all'.")
    return target

def initial_pks(target: str) -> tuple[list, Chem.Mol]:
    '''
    Computes a PKS design to synthesize the 2D structure of the target molecule

    Args:
        target (str): The SMILES string of the target molecule

    Returns:
        pks_design (list): Initial RetroTide design
        mol (Chem.Mol): The computed product of the PKS design
    '''
    designs = designPKS(Chem.MolFromSmiles(target),
                        maxDesignsPerRound = 200,
                        similarity = 'mcs_without_stereo')
    pks_design = designs[-1][0][0].modules
    mol = designs[-1][0][0].computeProduct(structureDB)
    return pks_design, mol

def module_mapping(pks_design: list) -> Chem.Mol:
    '''
    Maps atoms of the PKS product to each module in the PKS design.

    Args:
        pks_design (list): The RetroTide proposed PKS design

    Returns:
        mapped_product (Chem.Mol): An atom-module mapped PKS product
    '''
    cluster = bcs.Cluster(modules = pks_design)
    mapped_product = atommappks.create_atom_maps(cluster)[-1]
    return mapped_product

def find_ring_size(target: Chem.Mol) -> int:
    '''
    Finds the size of the largest ring in the target molecule.

    Args:
        target (Chem.Mol): The target molecule

    Returns:
        ring_size (int): The number of bonds in the largest ring found
    '''
    ring_info = target.GetRingInfo()
    
    if not ring_info.AtomRings():
        return 0
    
    # Calculates the number of atoms in the largest ring
    largest_ring = max(ring_info.AtomRings(), key=len)

    # Calculate the number of bonds in the largest ring
    ring_size = len(largest_ring) - 1
    return ring_size

def offload_pks_product(pks_product: Chem.Mol, target: Chem.Mol, pks_release_mechanism: str) -> tuple:
    '''
    Offloads the PKS product using the specified release mechanism.

    Args:
        pks_product (Chem.Mol): The PKS bound product
        target (Chem.Mol): The target molecule
        pks_release_mechanism (str): The mechanism to use for offloading the PKS product
            Options are 'thiolysis' or 'cyclization'

    Returns:
        unbound_mol (Chem.Mol): The unbound PKS product
    '''
    if pks_release_mechanism == 'thiolysis':
        Chem.SanitizeMol(pks_product)  
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
        products = rxn.RunReactants((pks_product,))

        if not products:
            print("Warning: No products generated from thiolysis reaction.")
            return pks_product,
    
        unbound_mol = products[0][0]
        Chem.SanitizeMol(unbound_mol)
        return (unbound_mol,)

    if pks_release_mechanism == 'cyclization':
        Chem.SanitizeMol(pks_product)
        target_ring_size = find_ring_size(target)

        print(f"Target ring size: {target_ring_size}")
        
        thioester_pattern = '[C:1](=[O:2])[S:3]'
        thioester_matches = pks_product.GetSubstructMatches(
            Chem.MolFromSmarts(thioester_pattern))
        
        carbonyl_carbon = thioester_matches[0][0]
        
        hydroxyl_pattern = '[OH1]'
        hydroxyl_matches = pks_product.GetSubstructMatches(
            Chem.MolFromSmarts(hydroxyl_pattern))

        if not hydroxyl_matches:
            print("Warning: No hydroxyl groups found")
            return (pks_product,)
        
        # Look for hydroxyl at the same distance as the target ring size
        for i, hydroxyl in enumerate(hydroxyl_matches):
            oxygen_idx = hydroxyl[0]
            
            # Calculate distance from the carbonyl carbon to this hydroxyl
            try:
                path = Chem.GetShortestPath(pks_product, carbonyl_carbon, oxygen_idx)
                distance = len(path) - 1
                
                # Proceed if distance matches target ring size
                if distance == target_ring_size:
                    # Manually cyclize the molecule
                    try:
                        editable_mol = Chem.EditableMol(pks_product)
                        
                        sulfur_idx = None
                        for atom in pks_product.GetAtoms():
                            if atom.GetSymbol() == 'S':
                                sulfur_idx = atom.GetIdx()
                                break
                        
                        if sulfur_idx is not None:
                            # Remove the C-S bond
                            editable_mol.RemoveBond(carbonyl_carbon,
                                                    sulfur_idx)
                            
                            # Add the C-O bond (cyclization)
                            editable_mol.AddBond(carbonyl_carbon,
                                                 oxygen_idx,
                                                 Chem.rdchem.BondType.SINGLE)
                            
                            # Remove the sulfur atom
                            editable_mol.RemoveAtom(sulfur_idx)
                            
                            unbound_mol = editable_mol.GetMol()
                            Chem.SanitizeMol(unbound_mol)
                            
                            print(f"Successfully cyclized at distance {distance}")
                            return (unbound_mol,)
                        else:
                            print("No sulfur atom found for removal")
                            
                    except Exception as e:
                        print(f"Failed to cyclize at distance {distance}: {e}")
                        
            except Exception as e:
                print(f"Could not calculate distance for hydroxyl {i+1}: {e}")
        
        # If no hydroxyl found at correct distance, return original molecule
        print(f"No hydroxyl found at target distance {target_ring_size}")
        return (pks_product,)

    return (unbound_mol,)

def matching_target_atoms(unbound_mol: Chem.Mol, target_mol: Chem.Mol) -> tuple[tuple, Chem.Mol]:
    '''
    Notes atom indices of the maximum common substructure between the unbound PKS product
    and the target molecule

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
        target_mol (Chem.Mol): The target molecule
    
    Returns:
        mol2_match (tuple): Atom indices in the target molecule that match the MCS
        mol2_copy (Chem.Mol): Copy of the target molecule with MCS highlighted
    '''
    mol1_copy = Chem.Mol(unbound_mol)
    mol2_copy = Chem.Mol(target_mol)
    mcs_no_chiral = rdFMCS.FindMCS(
        [mol1_copy, mol2_copy], 
        timeout=10, 
        matchValences=True, 
        matchChiralTag=False, 
        bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True)

    if mcs_no_chiral.numAtoms > 0:
        mcs_smarts = Chem.MolFromSmarts(mcs_no_chiral.smartsString)
        mol2_match = mol2_copy.GetSubstructMatch(mcs_smarts)
    else:
        mol2_match = tuple()
    return mol2_match, mol2_copy

def extract_target_substructure(target_mol: Chem.Mol, selected_atoms: list) -> Chem.Mol:
    """
    Remove unselected atoms and bonds from a molecule while preserving the original atom indices.
    """
    mol_copy = Chem.Mol(target_mol)

    # Store original atom indices
    for atom in mol_copy.GetAtoms():
        atom.SetIntProp("original_idx", atom.GetIdx())
    
    editable_mol = Chem.EditableMol(mol_copy)
    selected_set = set(selected_atoms)
    
    # Remove bonds between selected and unselected atoms
    bonds_to_remove = []
    for bond in target_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # If one atom is selected and other is not
        if (begin_idx in selected_set) != (end_idx in selected_set):
            bonds_to_remove.append((begin_idx, end_idx))
    
    # Remove bonds to unselected atoms
    for begin_idx, end_idx in bonds_to_remove:
        try:
            editable_mol.RemoveBond(begin_idx, end_idx)
        except:
            pass
    
    # Remove unselected atoms
    all_atoms = set(range(target_mol.GetNumAtoms()))
    atoms_to_remove = sorted(all_atoms - selected_set, reverse=True)
    
    for atom_idx in atoms_to_remove:
        editable_mol.RemoveAtom(atom_idx)
    
    result_mol = editable_mol.GetMol()
    
    # Sanitize with implicit hydrogens
    try:
        Chem.SanitizeMol(result_mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_VALENCE)
    except:
        try:
            Chem.SanitizeMol(result_mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES^Chem.SANITIZE_VALENCE)
        except:
            pass
    
    return result_mol

def mcs_map_product_to_target(unbound_mol: Chem.Mol, target_mol: Chem.Mol) -> pd.DataFrame:
    '''
    Uses maximum common substructure search to map atoms of the offloaded PKS product to the target molecule

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
        target_mol (Chem.Mol): The target molecule

    Returns:
        mcs_mapped_atoms_df (pd.DataFrame): DataFrame containing atom type, product atom index, and target atom index
    '''
    mol1_copy = Chem.Mol(unbound_mol)
    mol2_copy = Chem.Mol(target_mol)

    mcs_mapped_atoms_df = pd.DataFrame(columns=['Atom Type', 'Product Atom Idx', 'Target Atom Idx'])
    
    #Map product and target atoms based on structural correspondence
    mcs_no_chiral = rdFMCS.FindMCS([mol1_copy, mol2_copy], 
                                    timeout=10, 
                                    matchValences=True, 
                                    matchChiralTag=False, 
                                    bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
                                    ringMatchesRingOnly=False)

    if mcs_no_chiral.numAtoms > 0:
        mcs_smarts = Chem.MolFromSmarts(mcs_no_chiral.smartsString)

        mol1_match = mol1_copy.GetSubstructMatch(mcs_smarts)
        mol2_match = mol2_copy.GetSubstructMatch(mcs_smarts)

        for i, (prod_idx, target_idx) in enumerate(zip(mol1_match, mol2_match)):
            atom_symbol = mol1_copy.GetAtomWithIdx(prod_idx).GetSymbol()
            mcs_atom_entry = pd.DataFrame({
                'Atom Type': atom_symbol,
                'Product Atom Idx': prod_idx,
                'Target Atom Idx': target_idx}, index = [i])
            mcs_mapped_atoms_df = pd.concat([mcs_mapped_atoms_df, mcs_atom_entry], ignore_index=True)
    else:
        print("No common substructure found between the PKS product and target molecule.")
        return mcs_mapped_atoms_df
    return mcs_mapped_atoms_df

def map_product_to_pks_modules(unbound_mol: Chem.Mol) -> pd.DataFrame:
    '''
    Stores module mapping in a dataframe by extracting atom labels for the offloaded PKS product

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product

    Returns:
        atommapped_pks_df (pd.DataFrame): DataFrame containing atom type, product atom index, and module index
    '''
    atommapped_pks_df = pd.DataFrame(columns=['Atom Type', 'Product Atom Idx', 'Module'])
    for i, atom in enumerate(unbound_mol.GetAtoms()):
        if not atom.HasProp("atomLabel"):
            atom_label = '-'
        else:
            atom_label = atom.GetProp("atomLabel")

        atom_entry = pd.DataFrame({
            'Atom Type': [atom.GetSymbol()],
            'Product Atom Idx': [atom.GetIdx()],
            'Module': [atom_label]
        }, index = [i])
        atommapped_pks_df = pd.concat([atommapped_pks_df, atom_entry])
    return atommapped_pks_df

def get_full_mapping(mcs_map: pd.DataFrame, module_map: pd.DataFrame) -> pd.DataFrame:
    fully_mapped_molecule_df = pd.merge(mcs_map, module_map, on=['Atom Type', 'Product Atom Idx']).dropna()
    return fully_mapped_molecule_df

def check_chiral_centers(pks_product: Chem.Mol, target_mol: Chem.Mol, mapped_atoms: pd.DataFrame) -> tuple[list, list, list, list, dict, dict]:
    '''
    Checks chirality of mapped atoms between the PKS product and target molecule

    Returns:
        matching_atoms_1 (list): List of atom indices in PKS product with matching chirality
        matching_atoms_2 (list): List of atom indices in target molecule with matching chirality
        mismatching_atoms_1 (list): List of atom indices in PKS product with mismatching chirality
        mismatching_atoms_2 (list): List of atom indices in target molecule with mismatching chirality
        chiral_centers_1 (dict): Dictionary of chiral centers in PKS product with their chirality
        chiral_centers_2 (dict): Dictionary of chiral centers in target molecule with their chirality
    '''
    chiral_centers_1 = dict(Chem.FindMolChiralCenters(pks_product, includeUnassigned=True))
    chiral_centers_2 = dict(Chem.FindMolChiralCenters(target_mol, includeUnassigned=True))

    matching_atoms_1 = []
    matching_atoms_2 = []
    mismatching_atoms_1 = []
    mismatching_atoms_2 = []
    
    if len(mapped_atoms) > 0:
        # Extract atom indices from DataFrame
        mol1_match = mapped_atoms['Product Atom Idx'].astype(int).tolist()
        mol2_match = mapped_atoms['Target Atom Idx'].astype(int).tolist()
        
        print(f"Found {len(mol1_match)} atom correspondences from mapping DataFrame")
        
        # Compare chirality of mapped atoms
        for atom1_idx, atom2_idx in zip(mol1_match, mol2_match):
            # Check if both atoms are chiral
            if atom1_idx in chiral_centers_1 and atom2_idx in chiral_centers_2:
                chirality_1 = chiral_centers_1[atom1_idx]
                chirality_2 = chiral_centers_2[atom2_idx]
                
                if chirality_1 == chirality_2:
                    matching_atoms_1.append(atom1_idx)
                    matching_atoms_2.append(atom2_idx)
                else:
                    mismatching_atoms_1.append(atom1_idx)
                    mismatching_atoms_2.append(atom2_idx)
    return matching_atoms_1, matching_atoms_2, mismatching_atoms_1, mismatching_atoms_2, chiral_centers_1, chiral_centers_2

def plotMolComparison_with_mcs_mapping(mol1: Chem.Mol, mol2: Chem.Mol, 
                                       match1, match2, mmatch1, mmatch2,
                                       flip_mol1: bool, flip_direction: str,
                                       label1="PKS Product", label2="Target",
                                       chiral_centers_1=None, chiral_centers_2=None):
    """
    Visualize chiral centers and their correspondence between two molecules
    """
    # Create copies of the molecules to add atom labels
    mol1_copy = Chem.Mol(mol1)
    mol2_copy = Chem.Mol(mol2)

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol1_copy)
    AllChem.Compute2DCoords(mol2_copy)
    
    # Flip mol1 image if requested
    if flip_mol1:
        conf = mol1_copy.GetConformer()
        for i in range(mol1_copy.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            if flip_direction == 'horizontal':
                # Flip x-coordinate
                conf.SetAtomPosition(i, [-pos.x, pos.y, pos.z])
            elif flip_direction == 'vertical':
                # Flip y-coordinate
                conf.SetAtomPosition(i, [pos.x, -pos.y, pos.z])

    for atom in mol1_copy.GetAtoms():
        atom_idx = atom.GetIdx()
        base_label = f"{atom.GetSymbol()}:{atom_idx}"
        
        # Add R/S label if it's a chiral center
        if atom_idx in chiral_centers_1:
            chirality = chiral_centers_1[atom_idx]
            label = f"{base_label} ({chirality})"
        else:
            label = base_label
            
        atom.SetProp("atomNote", label)
    
    for atom in mol2_copy.GetAtoms():
        atom_idx = atom.GetIdx()
        base_label = f"{atom.GetSymbol()}:{atom_idx}"
        
        # Add R/S label if it's a chiral center
        if atom_idx in chiral_centers_2:
            chirality = chiral_centers_2[atom_idx]
            label = f"{base_label} ({chirality})"
        else:
            label = base_label
            
        atom.SetProp("atomNote", label)

    # Create highlight colors
    highlight_colors_1 = {}
    highlight_colors_2 = {}
    
    # Green for matching chirality
    for atom_idx in match1:
        highlight_colors_1[atom_idx] = (0, 1, 0)
    for atom_idx in match2:
        highlight_colors_2[atom_idx] = (0, 1, 0)

    # Red for mismatching chirality
    for atom_idx in mmatch1:
        highlight_colors_1[atom_idx] = (1, 0, 0)
    for atom_idx in mmatch2:
        highlight_colors_2[atom_idx] = (1, 0, 0)
    
    # All highlighted atoms
    all_highlighted_1 = match1 + mmatch1
    all_highlighted_2 = match2 + mmatch2

    print(f"\nFound {len(match1)} matching chiral centers")
    print(f"Found {len(mmatch1)} mismatching chiral centers")

    img = Draw.MolsToGridImage([
        mol1_copy, mol2_copy], 
        legends=[label1, label2], 
        molsPerRow=2,
        highlightAtomLists=[all_highlighted_1, all_highlighted_2],
        highlightAtomColors=[highlight_colors_1, highlight_colors_2],
        highlightBondLists=[[], []], 
        useSVG=True, 
        subImgSize=(500, 400))
    
    return img

def get_bcs_info(pks_design)-> dict:
    '''
    Constructs a dictionary containing information about the PKS design, including module number,
    substrate, and possible KR types.

    Args:
        pks_design (bcs object): Initial RetroTide design

    Returns:
        pks_features (dict): Dictionary with information about the intial PKS design
    '''
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
    '''
    Finds pairs of adjacent backbone carbon atoms in the PKS product
    '''
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
        return float('inf')

def filter_sequential_module_pairs(adjacent_pairs: list) -> list:
    '''
    Uses the module order to extract pairs of adjacent carbon atoms that are from sequential PKS modules
    '''
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
    '''
    Cross references each pairs with the list of mismatched chiral centers
    '''
    mismatch_pairs = []
    
    for atom1_idx, module1, atom2_idx, module2 in sequential_pairs:
        # Check if either atom is in the mismatch list
        if atom1_idx in mmatch1 or atom2_idx in mmatch1:
            mismatch_pairs.append((atom1_idx, module1, atom2_idx, module2))
    
    return mismatch_pairs

def check_substituent_patterns(mol: Chem.Mol, mismatch_pairs: list) -> list:
    #Use SMARTS patters to identify substituent patters around mismatched chiral centers
    alpha_carbon_patterns = {
        'alpha_to_carbonyl': '[C:1][C:2](=[O:3])',
        'alpha_to_hydroxyl': '[C:1][C:2][OH:3]',
        'alpha_to_ester': '[C:1][C:2](=[O:3])[O:4][C:5]' 
    }
    
    hydroxyl_carbon_pattern = '[C:1][OH:2]'
    
    # Pattern to detect ester formed via cyclization
    ester_oxygen_pattern = '[C:1](=[O:2])[O:3][C:4]' 
    
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
        alpha_atoms = set()
        
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
        
        # Check ester pattern - look for carbons bonded to ester-forming oxygen
        ester_pattern_mol = Chem.MolFromSmarts(ester_oxygen_pattern)
        ester_matches = mol.GetSubstructMatches(ester_pattern_mol)
        
        for match in ester_matches:
            ester_carbon_idx = match[3]
            
            if ester_carbon_idx == atom1_idx:
                pair_info['atom1_patterns'].append('ester_substituted')
            elif ester_carbon_idx == atom2_idx:
                pair_info['atom2_patterns'].append('ester_substituted')
        
        results.append(pair_info)
    
    return results

def kr_swaps(pks_features: dict, pattern_results: list, mmatch1: list):
    """
    Update KR types based on chiral mismatches and their structural patterns
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
            print(f"  Mismatched atom {mismatched_atom}: alpha={is_alpha}, beta={is_hydroxyl}")
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
    '''
    Reconstruct PKS design based on updated KR types

    Returns:
        pks_product (Chem.Mol): The final PKS product after stereocorrections
    '''
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
    return pks_product, modules

def check_similarity(final_corrected_prod: Chem.Mol, target_mol: Chem.Mol) -> float:
    '''
    Check Jaccard similarity between the final PKS product and the target molecule

    Returns:
        score (float): Jaccard similarity score between the two molecules
    '''
    fp_prod = fingerprints.get_fingerprint(Chem.MolToSmiles(final_corrected_prod), 'mapchiral')
    fp_targ = fingerprints.get_fingerprint(Chem.MolToSmiles(target_mol), 'mapchiral')

    score = similarity.get_similarity(fp_prod, fp_targ, 'jaccard')
    return score

def check_smiles_match(smiles1: str, smiles2: str) -> bool:
    if smiles1 == smiles2:
        return True
    else:
        return False

def main(target: str):
    '''
    Main function to perform post processing of a PKS design for a given target molecule.

    It maps the PKS product to each PKS module and the target molecule to identify mismatching
    chiral centers and which modules need a KR swap
    '''
    #Retrieve cannonical RDKit mol object for target molecule
    target_molecule = canonicalize_smiles(target, 'R/S')
    target_mol = Chem.MolFromSmiles(target_molecule)

    #Generate initial RetroTide PKS design
    pks_design, pks_product = initial_pks(target_molecule)
    pks_features = get_bcs_info(pks_design)

    #Map the PKS product to PKS modules
    mapped_product = module_mapping(pks_design)

    #Map the offloaded PKS product to the target molecule using MCS
    unbound_mol = offload_pks_product(mapped_product, target_mol, 'cyclization')[0]

    score = compareToTarget(unbound_mol, target_mol, similarity = 'mcs_without_stereo')

    if score < 1.0:
        print(f"Initial PKS product only matches the 2D target with {score*100:.3f}% MCS similarity")
        print("Extracting common substructure from target to assess chiral centers from")

        mol2_match, mol2_copy = matching_target_atoms(unbound_mol, target_mol)
        mol2_submol = extract_target_substructure(mol2_copy, list(mol2_match))

        mcs_mapped_atoms_df = mcs_map_product_to_target(unbound_mol, mol2_submol)
        target_mol = mol2_submol
    else:
        mcs_mapped_atoms_df = mcs_map_product_to_target(unbound_mol, target_mol)

    module_mapped_atoms_df = map_product_to_pks_modules(unbound_mol)
    full_mapping_df = get_full_mapping(mcs_mapped_atoms_df, module_mapped_atoms_df)

    #Store indices of matching and unmatching chiral centers between the PKS product and target molecule
    match1, match2, mmatch1, mmatch2, chiral_centers_1, chiral_centers_2 = check_chiral_centers(
        unbound_mol, target_mol, full_mapping_df)
    
    #Visualize chiral mismatches before making KR type modifications
    img = plotMolComparison_with_mcs_mapping(unbound_mol, target_mol, 
                                       match1, match2, mmatch1, mmatch2,
                                       flip_mol1=True, flip_direction='vertical',
                                       label1="PKS Product", label2="Target",
                                       chiral_centers_1=chiral_centers_1,
                                       chiral_centers_2=chiral_centers_2)   
    
    with open('mcs_mapping_image_pre.svg', 'w') as f:
        f.write(img)
    
    adjacent_pairs = find_adjacent_backbone_carbon_pairs(unbound_mol, full_mapping_df)
    sequential_pairs = filter_sequential_module_pairs(adjacent_pairs)

    pairs_with_mismatches = report_pairs_with_chiral_mismatches(sequential_pairs, mmatch1)
    pattern_results = check_substituent_patterns(unbound_mol, pairs_with_mismatches)

    pks_features_updated = kr_swaps(pks_features, pattern_results, mmatch1)

    final_modules = new_design(pks_features_updated)[1]

    mapped_final_prod = module_mapping(final_modules)
    final_corrected_prod = offload_pks_product(mapped_final_prod, target_mol, 'cyclization')[0]

    f_match1, f_match2, f_mmatch1, f_mmatch2, f_chiral_centers_1, f_chiral_centers_2 = check_chiral_centers(final_corrected_prod, target_mol, full_mapping_df)

    swaps_score = len(f_match1) / (len(f_match1) + len(f_mmatch1)) if (len(f_match1) + len(f_mmatch1)) > 0 else "N/A"
    print(f"Swaps Score: {swaps_score:.4f}")


    f_img = plotMolComparison_with_mcs_mapping(final_corrected_prod, target_mol,
                                   f_match1, f_match2, f_mmatch1, f_mmatch2,
                                   flip_mol1 = True, flip_direction = "vertical",
                                   chiral_centers_1=f_chiral_centers_1,
                                   chiral_centers_2=f_chiral_centers_2)
    
    with open('mcs_mapping_image_post.svg', 'w') as f:
        f.write(f_img)

    sim_score = check_similarity(final_corrected_prod, target_mol) #Perhaps use whole target not just the matching substructure
    
    print(f"Jaccard Similarity: {sim_score:.4f}")

    ground_truth = check_smiles_match(Chem.MolToSmiles(final_corrected_prod), Chem.MolToSmiles(target_mol))
    print(f"Do the corrected PKS product and target molecule/substructure match? {ground_truth}")

    return pks_design

if __name__ == "__main__":
    target = 'C[C@H]1C[C@H](C[C@@H]([C@H](/C(=C\C=C\C[C@H](OC(=O)C[C@@H]([C@H](C1)C)O)[C@@H]2CCC[C@H]2C(=O)O)/C#N)O)C)C'
    main(target)