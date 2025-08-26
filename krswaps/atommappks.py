import bcs
from typing import Optional, List
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdmolops
from typing import List, Tuple
from retrotide import structureDB

def find_farthest_carbon_from_sulfur(PKS_product: Chem.Mol) -> Tuple[int, int]:
    """
    Takes a PKS product and finds the carbon atom farthest from the sulfur radical, which is a placeholder for bond to ACP protein domain.
    
    Args:
        PKS_product(rdChem.Mol): a mol object representing the PKS product.
    Returns:
        Tuple[int, int]: the index of the carbon atom farthest from the PKS product as well as the corresponding bond distance
    """
    # first, we identify the sulfur atom in the PKS product
    sulfur_idx = None
    for atom in PKS_product.GetAtoms():
        if atom.GetSymbol() == "S":
            sulfur_idx = atom.GetIdx()
            break
    
    if sulfur_idx is None:
        raise ValueError("No sulfur atom found in the molecule.")
    
    # then, grab the atom indices of all carbon atoms in this PKS substrate
    carbon_indices = [atom.GetIdx() for atom in PKS_product.GetAtoms() if atom.GetSymbol() == "C"]
    
    if not carbon_indices:
        raise ValueError("No carbon atoms found in the molecule.")
    
    # compute the shortest path distances from sulfur
    distances = rdmolops.GetDistanceMatrix(PKS_product)
    
    # find the farthest carbon atom
    farthest_carbon_idx = max(carbon_indices, key = lambda idx: distances[sulfur_idx, idx])
    farthest_distance = distances[sulfur_idx, farthest_carbon_idx]
    
    return farthest_carbon_idx, farthest_distance

def add_fluorine_to_farthest_carbon(PKS_product: Chem.Mol, farthest_carbon_idx: int) -> Chem.Mol: 
    """
    Adds a fluorine atom to the carbon atom previously identified as farthest from the sulfur radical (which represents bond to ACP domain).
    
    Args:
        PKS_product(rdChem.Mol): a mol object representing the PKS product.
        farthest_carbon_idx(int): the internal RDKit atom index of the carbon that is the farthest away frm the sulfur radical.
        
    Returns:
        rdChem.Mol: the updated molecule with the fluorine atom added to the specified carbon atom.
    """

    editable_mol = Chem.EditableMol(PKS_product)
    fluorine_idx = editable_mol.AddAtom(Chem.Atom("F"))

    # add a bond between the specific carbon atom (e.g., index 1) and fluorine
    editable_mol.AddBond(farthest_carbon_idx, fluorine_idx, Chem.rdchem.BondType.SINGLE)

    PKS_product_w_F = editable_mol.GetMol()
    
    return PKS_product_w_F

def replace_fluorine_with_CoA(PKS_product_w_F: Chem.Mol) -> Chem.Mol:
    """
    Replaces the fluorine atom with a -CoA group to help in aligning molecules for downstream maximum common substructure comparisons.
    
    Args:
         PKS_product_w_F (rdChem.Mol): a mol object representing the PKS product with fluorine attached.
        
    Returns:
        rdChem.Mol: a mol object representing the PKS product with -CoA attached instead of fluorine.
    """
    
    attach_CoA_rxn_pattern = '[C,c:1]-[F:2].[S:3]>>[C,c:1]-[S:3].[F:2]'
    attach_CoA_rxn = AllChem.ReactionFromSmarts(attach_CoA_rxn_pattern)
    
    CoA_group = Chem.MolFromSmiles('CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCS)O')
    
    for atom in CoA_group.GetAtoms():
        atom.SetProp("atomLabel","X")
        
    PKS_product_w_CoA = attach_CoA_rxn.RunReactants((PKS_product_w_F, CoA_group))[0][0]
    return PKS_product_w_CoA

def detach_CoA_frm_labelled_PKS_product(PKS_product_w_CoA: Chem.Mol) -> Chem.Mol:
    """
    Detaches the -CoA group from the carbon atom that is farthest from the sulfur radical to regenerate the original PKS product.
    
    Args:
        PKS_product_w_CoA(rdChem.Mol): a mol object representing the PKS product with a -CoA group.
        
    Returns:
        rdchem.Mol: a mol object representing the original PKS product without the -CoA group attached.
    """
    
    detach_CoA_rxn_pattern = '[C,c:1]-[S:3]-[C:4]-[C:5]-[N:6]>>[C,c:1].[S:3]-[C:4]-[C:5]-[N:6]'
    detach_CoA_rxn = AllChem.ReactionFromSmarts(detach_CoA_rxn_pattern)
    labelled_PKS_product = detach_CoA_rxn.RunReactants((PKS_product_w_CoA,))[0][0]
    return labelled_PKS_product

def create_atom_maps(cluster: bcs.Cluster) -> Chem.Mol:
    
    modules_list = cluster.modules
    atom_labels_list = [f"LM" if i == 0 else f"M{i}" for i in range(len(modules_list))]
    labelled_mols_list = []
    
    for module_num in range(0,len(modules_list)):
        
        # for the loading module specifically
        if module_num == 0:
            
            # generate the loading module product
            cluster = bcs.Cluster(modules = [ modules_list[0] ]) # input to bcs.Cluster is list of modules
            LM_product = cluster.computeProduct(structureDB)
            
            # attach a florine atom to the farthest carbon from the sulfur radical (ACP placeholder)
            farthest_carbon_idx, _ = find_farthest_carbon_from_sulfur(LM_product)
            LM_product_w_F = add_fluorine_to_farthest_carbon(LM_product, farthest_carbon_idx)
            
            # then swap the florine for a -CoA group so all PKS products can be easily aligned
            LM_product_w_CoA = replace_fluorine_with_CoA(LM_product_w_F)
            
            for atom in LM_product_w_CoA.GetAtoms():
                if atom.HasProp("atomLabel") == 0:
                    atom.SetProp("atomLabel","LM")
                    
            labelled_mols_list.append(LM_product_w_CoA)
            
        else:
            
            current_modules_sublist = modules_list[0:(module_num+1)]
            atom_labels_sublist = atom_labels_list[0:(module_num+1)]
            current_cluster = bcs.Cluster(modules = current_modules_sublist)
            current_product = current_cluster.computeProduct(structureDB)
            
            # similar to the loading module, attach a fluorine to the farthest carbon atom from ACP
            farthest_carbon_idx, _ = find_farthest_carbon_from_sulfur(current_product)
            current_product_w_F = add_fluorine_to_farthest_carbon(current_product, farthest_carbon_idx)
            current_product_w_CoA = replace_fluorine_with_CoA(current_product_w_F)
            
            # for each previous_product starting from up until the previous module
            for i, previous_product_w_CoA in enumerate(labelled_mols_list[::-1]):
                
                MCS_result = rdFMCS.FindMCS([current_product_w_CoA, previous_product_w_CoA], 
                                            timeout = 1,
                                            matchChiralTag = False,
                                            bondCompare = Chem.rdFMCS.BondCompare.CompareOrderExact)
                
                match = current_product_w_CoA.GetSubstructMatch(MCS_result.queryMol)
                
                label = atom_labels_sublist[::-1][i]
                
                for atom in current_product_w_CoA.GetAtoms():
                    if atom.GetIdx() not in match:
                        if atom.HasProp("atomLabel") == 0:
                            atom.SetProp("atomLabel", label)
                            
            labelled_mols_list.append(current_product_w_CoA)
            
    # for all the labelled PKS substrates, detach the -CoA group
    final_mols_list = [detach_CoA_frm_labelled_PKS_product(mol) for mol in labelled_mols_list]
    
    # finally, label atoms that were previously attached to the -CoA group with "LM"
    for mol in final_mols_list:
        for atom in mol.GetAtoms():
            if atom.HasProp("atomLabel")==0:
                atom.SetProp("atomLabel","LM")
                
    return final_mols_list