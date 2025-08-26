"""Module for processing PKS products before stereo correction."""
# pylint: disable=no-member
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
import pandas as pd
from retrotide import structureDB, designPKS, compareToTarget
import bcs
from krswaps import atommappks

def canonicalize_smiles(molecule_str: str, stereo: str) -> str:
    """
    Standardizes a SMILES string and returns its canonical form with the
    specified isomeric information.
    
    Args:
        molecule_str (str): The SMILES string of the target molecule
        stereo (str): The type of stereochemistry to specify

    Returns:
        str: The canonicalized SMILES string.
    """
    mol = Chem.MolFromSmiles(molecule_str)
    if stereo == 'R/S':
        target = Chem.MolToSmiles(mol, isomericSmiles=True).replace('/', '').replace('\\', '')
    elif stereo == 'E/Z':
        target = Chem.MolToSmiles(mol, isomericSmiles=True).replace('@', '').replace('@@', '')
    elif stereo == 'none':
        target = Chem.MolToSmiles(mol, isomericSmiles=False)
    elif stereo == 'all':
        target = Chem.MolToSmiles(mol, isomericSmiles=True) 
    else:
        raise ValueError("Invalid stereo option. Choose from 'R/S', 'E/Z', 'none', or 'all'.")
    return target

def initial_pks(target: str) -> tuple[list, Chem.Mol]:
    """
    Computes a PKS design to synthesize the 2D structure of the target molecule

    Args:
        target (str): The SMILES string of the target molecule

    Returns:
        pks_design (list): Initial RetroTide design
        mol (Chem.Mol): The computed product of the PKS design
    """
    designs = designPKS(Chem.MolFromSmiles(target),
                        maxDesignsPerRound = 200,
                        similarity = 'mcs_without_stereo')
    pks_design = designs[-1][0][0].modules
    mol = designs[-1][0][0].computeProduct(structureDB)
    return pks_design, mol

def target_ring_size(target_mol: Chem.Mol) -> int:
    """
    Determines the size of the largest ring in the target molecule.
    """
    ring_info = target_mol.GetRingInfo()
    if not ring_info.AtomRings():
        return 0
    largest_ring = max(ring_info.AtomRings(), key=len)
    target_size = len(largest_ring) - 1
    return target_size

def offload_pks_product(pks_product: Chem.Mol,
                        target_mol: Chem.Mol,
                        pks_release_mechanism: str) -> tuple:
    """
    Offloads the PKS product using the specified release mechanism.

    Args:
        pks_product (Chem.Mol): The PKS bound product
        target (Chem.Mol): The target molecule
        pks_release_mechanism (str): The mechanism to use for offloading the PKS product
            Options are 'thiolysis' or 'cyclization'

    Returns:
        unbound_mol (tuple): The unbound PKS product
    """
    thioester_pattern = '[C:1](=[O:2])[S:3]'
    thioester_matches = pks_product.GetSubstructMatches(Chem.MolFromSmarts(thioester_pattern))
    
    if pks_release_mechanism == 'thiolysis':
        Chem.SanitizeMol(pks_product)  
        
        c_idx, o_idx, s_idx = thioester_matches[0]
        if not thioester_matches:
            print("Warning: No products generated from thiolysis reaction.")
            return (pks_product,)
        
        editable_mol = Chem.EditableMol(pks_product)
        editable_mol.RemoveBond(c_idx, s_idx)
        editable_mol.RemoveAtom(s_idx)
        new_o_idx = editable_mol.AddAtom(Chem.Atom("O"))
        editable_mol.AddBond(c_idx, new_o_idx, Chem.rdchem.BondType.SINGLE)
        unbound_mol = editable_mol.GetMol()
        Chem.SanitizeMol(unbound_mol)
        return (unbound_mol,)

    if pks_release_mechanism == 'cyclization':
        Chem.SanitizeMol(pks_product)
        target_size = target_ring_size(target_mol)
        
        if not thioester_matches:
            print("Warning: No thioester found in PKS product")
            return (pks_product,)
        
        carbonyl_carbon = thioester_matches[0][0]
        hydroxyl_pattern = '[OH1]'
        hydroxyl_matches = pks_product.GetSubstructMatches(
            Chem.MolFromSmarts(hydroxyl_pattern))
        if not hydroxyl_matches:
            print("Warning: No hydroxyl groups found")
            return (pks_product,)
        for hydroxyl in hydroxyl_matches:
            oxygen_idx = hydroxyl[0]
            path = Chem.GetShortestPath(pks_product, carbonyl_carbon, oxygen_idx)
            distance = len(path) - 1
            if distance == target_size:
                editable_mol = Chem.EditableMol(pks_product)
                sulfur_idx = None
                for atom in pks_product.GetAtoms():
                    if atom.GetSymbol() == 'S':
                        sulfur_idx = atom.GetIdx()
                        break
                if sulfur_idx is not None:
                    editable_mol.RemoveBond(carbonyl_carbon,
                                            sulfur_idx)
                    editable_mol.AddBond(carbonyl_carbon,
                                            oxygen_idx,
                                            Chem.rdchem.BondType.SINGLE)
                    editable_mol.RemoveAtom(sulfur_idx)
                    unbound_mol = editable_mol.GetMol()
                    Chem.SanitizeMol(unbound_mol)
                    return (unbound_mol,)
        print(f"No hydroxyl found at target distance {target_size}")
        return (pks_product,)
    return (unbound_mol,)

def module_mapping(pks_design: list) -> Chem.Mol:
    """
    Maps atoms of the PKS product to each module in the PKS design.

    Args:
        pks_design (list): The RetroTide proposed PKS design

    Returns:
        mapped_product (Chem.Mol): An atom-module mapped PKS product
    """
    cluster = bcs.Cluster(modules = pks_design)
    mapped_product = atommappks.create_atom_maps(cluster)[-1]
    return mapped_product

def matching_target_atoms(unbound_mol: Chem.Mol, target_mol: Chem.Mol) -> tuple[tuple, Chem.Mol]:
    """
    Notes atom indices of the maximum common substructure between the unbound PKS product
    and the target molecule

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
        target_mol (Chem.Mol): The target molecule
    
    Returns:
        mol2_match (tuple): Atom indices in the target molecule that match the MCS
        mol2_copy (Chem.Mol): Copy of the target molecule with MCS highlighted
    """
    mcs_no_chiral = rdFMCS.FindMCS(
        [unbound_mol, target_mol], 
        timeout=10, 
        matchValences=True, 
        matchChiralTag=False, 
        bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=True)
    if mcs_no_chiral.numAtoms > 0:
        mcs_smarts = Chem.MolFromSmarts(mcs_no_chiral.smartsString)
        mol2_match = target_mol.GetSubstructMatch(mcs_smarts)
    else:
        mol2_match = tuple()
    return mol2_match, target_mol

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
        editable_mol.RemoveBond(begin_idx, end_idx)
    # Remove unselected atoms
    all_atoms = set(range(target_mol.GetNumAtoms()))
    atoms_to_remove = sorted(all_atoms - selected_set, reverse=True)
    for atom_idx in atoms_to_remove:
        editable_mol.RemoveAtom(atom_idx)
    result_mol = editable_mol.GetMol()
    try:
        Chem.SanitizeMol(result_mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_VALENCE)
    except Exception:
        try:
            Chem.SanitizeMol(result_mol,
                             sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES^Chem.SANITIZE_VALENCE)
        except Exception:
            pass
    return result_mol

def map_product_to_target(unbound_mol: Chem.Mol, target_mol: Chem.Mol) -> pd.DataFrame:
    """"
    Uses maximum common substructure search to map atoms of the offloaded
    PKS product to the target molecule

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product
        target_mol (Chem.Mol): The target molecule

    Returns:
        mcs_mapped_atoms_df (pd.DataFrame): DataFrame containing atom type,
        product atom index, and target atom index
    """
    mcs_mapped_atoms_df = pd.DataFrame(columns=['Atom Type', 'Product Atom Idx', 'Target Atom Idx'])
    mcs_no_chiral = rdFMCS.FindMCS([unbound_mol, target_mol], 
                                    timeout=10, 
                                    matchValences=True, 
                                    matchChiralTag=False, 
                                    bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact,
                                    ringMatchesRingOnly=False)
    if mcs_no_chiral.numAtoms > 0:
        mcs_smarts = Chem.MolFromSmarts(mcs_no_chiral.smartsString)
        mol1_match = unbound_mol.GetSubstructMatch(mcs_smarts)
        mol2_match = target_mol.GetSubstructMatch(mcs_smarts)
        for i, (prod_idx, target_idx) in enumerate(zip(mol1_match, mol2_match)):
            atom_symbol = unbound_mol.GetAtomWithIdx(prod_idx).GetSymbol()
            mcs_atom_entry = pd.DataFrame({
                'Atom Type': atom_symbol,
                'Product Atom Idx': prod_idx,
                'Target Atom Idx': target_idx}, index = [i])
            mcs_mapped_atoms_df = pd.concat([mcs_mapped_atoms_df, mcs_atom_entry],
                                            ignore_index=True)
    else:
        print("No common substructure found between the PKS product and target molecule.")
        return mcs_mapped_atoms_df
    return mcs_mapped_atoms_df

def map_product_to_pks_modules(unbound_mol: Chem.Mol) -> pd.DataFrame:
    """
    Stores module mapping in a dataframe by extracting atom labels for the offloaded PKS product

    Args:
        unbound_mol (Chem.Mol): The unbound PKS product

    Returns:
        atommapped_pks_df (pd.DataFrame): DataFrame containing atom type, product atom index,
        and module index
    """
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

def full_mapping(mcs_map: pd.DataFrame, module_map: pd.DataFrame) -> pd.DataFrame:
    """ Combines MCS mapping and module mapping into a single DataFrame."""
    fully_mapped_molecule_df = pd.merge(
        mcs_map, module_map, on=['Atom Type', 'Product Atom Idx']).dropna()
    return fully_mapped_molecule_df

ChiralCheckResult = namedtuple('ChiralCheckResult',
                               ['match1', 'match2', 'mmatch1', 'mmatch2', 'cc1', 'cc2'])
def check_chiral_centers(pks_product: Chem.Mol,
                         target_mol: Chem.Mol,
                         mapped_atoms: pd.DataFrame) -> ChiralCheckResult:
    """
    Checks chirality of mapped atoms between the PKS product and target molecule

    Returns:
        ChiralCheckResult: A named tuple containing lists of matching and mismatching atom indices
        and dicts of chiral centers
    """
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
    return ChiralCheckResult(matching_atoms_1, matching_atoms_2,
                             mismatching_atoms_1, mismatching_atoms_2,
                             chiral_centers_1, chiral_centers_2)

def initialize_pks_product(target, offload_mech):
    """
    Run RetroTide on the target molecule, module map its product and offload
    the product from the PKS
    """
    target_mol = Chem.MolFromSmiles(target)
    pks_design = initial_pks(target)[0]
    mapped_product = module_mapping(pks_design)
    unbound_product = offload_pks_product(mapped_product, target_mol, offload_mech)[0]
    return target_mol, pks_design, unbound_product

def add_atom_labels(mol: Chem.Mol, chiral_centers: dict) -> Chem.Mol:
    """
    Add atom labels to the molecule for visualization
    """
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        base_label = f"{atom.GetSymbol()}:{atom_idx}"
        # Add R/S label if it's a chiral center
        if atom_idx in chiral_centers:
            chirality = chiral_centers[atom_idx]
            label = f"{base_label} ({chirality})"
        else:
            label = base_label
        atom.SetProp("atomNote", label)
    return mol

def check_mcs(unbound_product, target_mol):
    """
    Assess 2D similarity using MCS. If < 1.0, extract common substructure and set as the target
    """
    mcs_score = compareToTarget(unbound_product, target_mol, similarity='mcs_without_stereo')
    if mcs_score < 1.0:
        print(f"Initial PKS product only matches {mcs_score*100:.1f}% of the 2D target")
        print("Extracting common substructure from target to assess chiral centers from")
        mol2_match, mol2_copy = matching_target_atoms(unbound_product, target_mol)
        mol2_submol = extract_target_substructure(mol2_copy, list(mol2_match))

        mcs_mapped_atoms_df = map_product_to_target(unbound_product, mol2_submol)
        return mol2_submol, mcs_mapped_atoms_df
    else:
        mcs_mapped_atoms_df = map_product_to_target(unbound_product, target_mol)
        return target_mol, mcs_mapped_atoms_df
