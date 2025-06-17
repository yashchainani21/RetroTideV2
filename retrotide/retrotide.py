# -*- coding: utf-8 -*-
"""
This file contains functions for designing Polyketide Synthases (PKS) modules aimed at producing molecules
similar to a given target structure, and for comparing molecular structures using various similarity metrics. 

The core functionality revolves around iterative design of PKS modules, leveraging a combinatorial approach to 
synthesize compounds closely matching the target molecule's structure. This involves two main functions: 
`compareToTarget` for assessing the similarity between two molecular structures, and `designPKS` for generating 
PKS designs iteratively until a design most closely resembling the target molecule is achieved.
The functionality is designed for use in synthetic biology and metabolic engineering research, particularly in 
the design of biosynthetic pathways for the production of polyketide-based compounds. 

Example:
    target_molecule = Chem.MolFromSmiles('CCO')
    final_designs = designPKS(target_molecule)
    print(final_designs[-1][0][1]) # Prints the best score from the final round
    best_pks = final_designs[-1][0][0] # Retrieves the best PKS design from the final round

@authors: Tyler Backman, Vincent Blay, Yash Chainani
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdFMCS
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.rdchem import Mol
import itertools
from typing import Union, Callable, List, Optional, Any, Tuple

from .extras import allStarterTypes, allModuleTypes, structureDB
from bcs import Cluster, Module
from .AtomAtomPathSimilarity import getpathintegers, AtomAtomPathSimilarity
from mapchiral.mapchiral import encode, jaccard_similarity

##########################################
#  helper functions for subgraph pruning #
##########################################

def run_pks_release_reaction(pks_release_mechanism: str,
                             bound_product_mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Run a PKS offloading reaction to release a PKS product bound to its synthase via either a thioreductase, thiolysis or a cyclization reaction

    Args:
        pks_release_mechanism (str): the offloading reaction to use when release a bound PKS product; choose from 'thiolysis', 'cyclization', and 'reduction'.
        bound_product_mol (Chem.Mol): mol object representing the bound PKS product or intermediate.

    Returns:
        Chem.Mol: mol object representing the released PKS product or intermediate.
        None: or None returned if the termination reaction is not possible (typically occurs with cyclization reactions)
    """
    if pks_release_mechanism == "thiolysis":
        Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to produce terminal acid group
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
        unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
        Chem.SanitizeMol(unbound_product_mol)
        return unbound_product_mol

    if pks_release_mechanism == "cyclization":
        Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to cyclize bound substrate
        rxn = AllChem.ReactionFromSmarts('([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]')
        try:
            unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
            Chem.SanitizeMol(unbound_product_mol)
            return unbound_product_mol

        # if the bound substrate cannot be cyclized, then return None
        except:
            return None

    if pks_release_mechanism == "reduction":
        Chem.SanitizeMol(bound_product_mol)  # run detachment reaction to fully reduce the bound substrate
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1]')
        unbound_product_mol = rxn.RunReactants((bound_product_mol,))[0][0]
        Chem.SanitizeMol(unbound_product_mol)
        return unbound_product_mol

def getSubmolRadN(mol: Chem.Mol,
                  radius: int) -> List[Chem.Mol]:
    """
    Iterate over all atoms in a molecule to collect subgraphs of radius N that can be extracted around each atom.

    Args:
        mol (Chem.Mol): mol object of an input molecule from which to extract subgraphs.
        radius (int): radius or number of bonds around an atom from which subgraphs should be extracted.

    Returns:
        List[Chem.Mol]: list of mol objects representing all extracted subgraphs.
    """
    
    # initialize an empty list to store all subgraphs
    submols = []
    
    atoms = mol.GetAtoms()

    for atom in atoms:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        subsmi = Chem.MolToSmiles(submol, rootedAtAtom=amap[atom.GetIdx()], canonical=False)
        submols.append(Chem.MolFromSmiles(subsmi, sanitize=False))
    return submols

def are_isomorphic(mol1: Chem.Mol,
                   mol2: Chem.Mol,
                   consider_stereo: bool = False) -> bool:
    """
    Compare two molecules to determine if they are equivalent and identical.

    Args:
        mol1 (Chem.Mol): mol object of first molecule.
        mol2 (Chem.Mol): mol object of second molecule.

    Returns:
        bool: returns True if both molecules are identical and False if otherwise.
    """
    
    if consider_stereo:
        is_isomorphic = mol1.HasSubstructMatch(mol2, useChirality=True) and mol2.HasSubstructMatch(mol1,
                                                                                                       useChirality=True)
    else:
        is_isomorphic = mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)

    return is_isomorphic

def create_bag_of_graphs_from_target(target: Chem.Mol) -> List[Chem.Mol]:
    """
    Create a bag of subgraphs from the target molecule. 

    Args:
        target (Chem.Mol): mol object of the target molecule.

    Returns:
        List[Chem.Mol]: list of mol objects representing subgraphs created from the target molecule.
    """

    # first, we get the longest distance between any two atoms within the target molecule
    dist_matrix = rdmolops.GetDistanceMatrix(target)
    dist_array = np.array(dist_matrix)
    longest_distance = dist_array.max()

    # using this longest distance, create a bag of graphs by decomposing the target molecule across various lengths
    all_submols = []
    for i in range(1, int(longest_distance + 1)):
        try:
            submols = getSubmolRadN(mol = target, radius = i)
            all_submols.extend(submols)
        except:
            pass

    # next, check if the input target is a cyclic ester (lactone)
    lactone_group = Chem.MolFromSmarts('[C;R](=O)[O;R]')
    if target.HasSubstructMatch(lactone_group):

        # if it is, then we run a ring opening reaction to simulate the action of a cyclic TE acting in reverse
        ring_opening_rxn_pattern = '[C:1](=[O:2])[*:4][C:5][C:6]>>([C:1](=[O:2])[O:3].[O,N:4][C:5][C:6])'
        ring_opening_rxn = AllChem.ReactionFromSmarts(ring_opening_rxn_pattern)
        ring_opened_target = ring_opening_rxn.RunReactants((target,))[0][0]

        # then, run the subgraph collection algorithm again on the ring opened target
        dist_matrix = rdmolops.GetDistanceMatrix(ring_opened_target)
        dist_array = np.array(dist_matrix)
        longest_distance = dist_array.max()

        for i in range(1, int(longest_distance + 1)):
            try:
                submols = getSubmolRadN(mol = ring_opened_target, radius = i)
                all_submols.extend(submols)
            except:
                pass

    # do nothing if the input target is not a cyclic ester
    else:
        pass

    return all_submols

def is_PKS_product_in_bag_of_graphs(bag_of_graphs: List[Chem.Mol],
                                    PKS_product: Chem.Mol,
                                    consider_stereo: bool = False) -> bool:
    """
    Check if a PKS product or intermediate that is still bound to its synthase is present in the bag of subgraphs generated from the final target molecule.
    Given an input PKS product or intermediate, this molecule is first released from its bound state by running all three possible offloading reactions.
    Then, we check if this unbound 

    Args:
        bag_of_graphs (List[Chem.Mol]): bag of subgraphs obtained from iterating over all atoms in the final target molecule.
        PKS_product (Chem.Mol): mol object of the 
    """

    acid_product = run_pks_release_reaction(pks_release_mechanism = 'thiolysis', bound_product_mol = PKS_product)
    fully_reduced_product = run_pks_release_reaction(pks_release_mechanism = 'reduction', bound_product_mol = PKS_product)
    cyclized_product = run_pks_release_reaction(pks_release_mechanism = 'cyclization', bound_product_mol = PKS_product)

    # iterate over all subgraphs derived from the target and stored within bag_of_graphs
    for submol in bag_of_graphs:

        # check if a given subgraph is equivalent to the acid product obtained from offloading
        if are_isomorphic(acid_product, submol):
            
            # terminate early if acid product is a full subgraph of the target molecule
            return True 
        
        # check if a given subgraph is equivalent to the fully reduced product obtained from offloading
        if are_isomorphic(fully_reduced_product, submol):

            # terminate early if the fully reduced product is a full subgraph of the target molecule
            return True

        # check if a cyclized product has formed
        if cyclized_product:

            # now check if a given subgraph is equivalent to the cyclized product obtained from offloading
            if are_isomorphic(cyclized_product, submol):

                # terminate early if the cyclized product is a full subgraph of the target molecule
                return True

    # return False if no match is found
    return False


#######################
# main RetroTide code #
#######################

def compareToTarget(structure: Mol,
                    target: Mol,
                    similarity: Union[str, Callable] = 'atompairs',
                    targetpathintegers = None,
                    bag_of_graphs: Optional[List[Chem.Mol]] = None) -> float:
    """
    Compares a given structure to a target molecule to determine their similarity.

    This function supports multiple similarity metrics, including atom pairs, maximum common substructure (MCS),
    and custom callable metrics.

    Args:
        structure (rdchem.Mol): The molecule to compare against the target.
        target (rdchem.Mol): The target molecule for comparison.
        similarity (Union[str, Callable], optional): The type of similarity metric to use. Can be 'mcs' for maximum
            common substructure, 'atompairs' for atom pair similarity, 'atomatompath' for atom-atom path similarity,
            'MAPC' for Min-hashed Atom Pair Chiral Fingerprints, 'subgraph_modified_mcs' for subgraph modified mcs,
            or a custom callable function taking two rdchem.Mol objects and returning a similarity score as a float.
            Defaults to 'atompairs'.

    Returns:
        float: The similarity score between the given structure and target. Higher scores indicate greater similarity.

    Raises:
        IOError: If an invalid similarity metric is specified.
    """    
    # convert to smiles and back to fix stereochem- this shouldn't be needed!
    # testProduct = Chem.MolFromSmiles(Chem.MolToSmiles(testProduct, isomericSmiles=True))

    # remove C(=O)S from testProduct before comparison
    testProduct = Chem.rdmolops.ReplaceSubstructs(structure, Chem.MolFromSmiles('C(=O)S'), Chem.MolFromSmiles('C'))[0]
    
    if similarity == 'mcs':
        # MCS
        result=rdFMCS.FindMCS([target, testProduct], timeout=1, matchValences=True, matchChiralTag=True, 
                              bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact) # search for 1 second max

        if result.canceled:
            print('MCS timeout')
        score = result.numAtoms / (testProduct.GetNumAtoms() + target.GetNumAtoms() - result.numAtoms)
    
    elif similarity == 'mcs_without_stereo':
        # same as above but without matching stereochemistry
        result=rdFMCS.FindMCS([target, testProduct], timeout=1, matchValences=True, matchChiralTag=False,
                              bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact) # search for 1 second max

        if result.canceled:
            print('MCS timeout')
        score = result.numAtoms / (len(testProduct.GetAtoms()) + len(target.GetAtoms()) - result.numAtoms)

    elif similarity == 'atompairs':
        # atom pair
        ms = [target, testProduct]
        pairFps = [Pairs.GetAtomPairFingerprint(x) for x in ms]
        score = DataStructs.TanimotoSimilarity(pairFps[0],pairFps[1]) # can also try DiceSimilarity
        
    elif similarity == 'atomatompath':
        score = AtomAtomPathSimilarity(target, testProduct, m1pathintegers=targetpathintegers)

    elif similarity == 'MAPC': # Min-hashed Atom Pair Chiral Fingerprints (https://github.com/reymond-group/mapchiral)

        # compute chiral fingerprints of target and testProduct then get their jaccard similarity
        target_fp = encode(target, max_radius=2, n_permutations=2048, mapping=False)
        testProduct_fp = encode(testProduct, max_radius=2, n_permutations=2048, mapping=False)
        score = jaccard_similarity(target_fp, testProduct_fp)

    elif similarity == 'subgraph_modified_mcs': 

        # here, check if an intermediate is in the bag of graphs created from the target molecule

        if is_PKS_product_in_bag_of_graphs(bag_of_graphs = bag_of_graphs, 
                                           PKS_product = structure):

            # if the PKS product/ intermediate is a full subgraph of the target molecule, 
            # then we calculate an MCS score so that the recursive call to RetroTide has a way to rank intermediate PKS designs
            result=rdFMCS.FindMCS([target, testProduct], timeout=1, matchValences=True, matchChiralTag=False,
                              bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact) # search for 1 second max

            if result.canceled:
                print('MCS timeout')
            score = result.numAtoms / (len(testProduct.GetAtoms()) + len(target.GetAtoms()) - result.numAtoms)
        
        # if an intermediate is not in the bag of graphs, then we assign it a score of 0 to eliminate the associated PKS design
        else:
            score = 0

    elif callable(similarity):
        score = similarity(target, testProduct)
    
    else:
        raise IOError('Invalid similarity input')
    
    return score

def designPKS(targetMol: Mol,
              previousDesigns: Optional[List[Tuple[Cluster, float, Mol]]] = None, 
              maxDesignsPerRound: int = 25,
              similarity: str = 'mcs_without_stereo') -> List[List[Any]]:
    """
    Designs polyketide synthases (PKS) modules to match a target molecule.

    This function iteratively designs PKS modules, comparing each design to the target molecule using a specified
    similarity metric. It supports multiple rounds of design, optionally starting from a set of previous designs,
    and limits the number of designs per round to manage computational complexity.

    Args:
        targetMol (Mol): The target molecule for the PKS design.
        previousDesigns (Optional[List[List[Any]]], optional): A list of previous design rounds to continue from.
            If None, the function starts a new design process. Defaults to None.
        maxDesignsPerRound (int, optional): The maximum number of designs to consider per round. Defaults to 25.
        similarity (str, optional): The similarity metric to use for comparing designs to the target molecule.
            Supported values are 'atompairs', atomatompath', 'mcs', 'mcs_without_stereo', and 'MAPC'. Defaults to 'mcs_without_stereo'.

    Returns:
        List[List[Any]]: A list of design rounds, where each round is a list of designs. Each design includes the
        PKS module configuration, its similarity score to the target molecule, and the resulting molecule. 
        Design rounds are in order of increasing iteration. Within each round, we sort the designs by score (descending).

    Note:
        - The function uses global variables for module types and starter types, which should be set before calling.
        - The actual implementation of PKS module design, including the computation of similarity scores and the
          generation of resulting molecules, is abstracted away in this description.

    Example:
        target_molecule = Chem.MolFromSmiles('CCO')
        final_designs = designPKS(target_molecule)
        print(final_designs[-1][0][1]) # Best score from the final round
        best_pks = final_designs[-1][0][0]) # Best PKS from the final round
    """

    # always create a bag of graphs by default even if the similarity metrics other than subgraph_similarity are used
    bag_of_graphs = create_bag_of_graphs_from_target(targetMol)

    targetpathintegers = None
    if previousDesigns is not None:
        print('computing module ' + str(len(previousDesigns)))
    else:
        print('computing module 1')
        if similarity=='atomatompath':
            targetpathintegers = getpathintegers(targetMol)

    if not previousDesigns:
        initial_designs = []
        for starter in allStarterTypes:
            cluster = Cluster(modules=[starter])
            product = cluster.computeProduct(structureDB)
            initial_designs.append((cluster, 0.0, product))
        previousDesigns = [initial_designs]

    # Combine the last round of designs with all module types to create extended sets
    last_round_designs = previousDesigns[-1]
    extendedSets = list(itertools.product(last_round_designs, allModuleTypes))

    # perform each extension
    designs: List[Cluster] = [Cluster(modules=x[0][0].modules + [x[1]]) for x in extendedSets]
    
    print('   testing ' + str(len(designs)) + ' designs')
    
    # compute structures
    prevStructures: List[Mol] = [x[0][2] for x in extendedSets]
    
    structures: List[Mol] = [design.computeProduct(structureDB, chain=prevStructure) for design, prevStructure in zip(designs, prevStructures)]

    # compare modules to target
    scores: List[float] = [compareToTarget(structure, targetMol, similarity, targetpathintegers, bag_of_graphs) for structure in structures]
    
    # assemble scores
    assembledScores: List[Tuple[Cluster, float, Mol]] = list(zip(designs, scores, structures))
    
    # sort designs by score
    assembledScores.sort(reverse=True, key=lambda x: x[1])
      
    # find best score from previous design round
    bestPreviousScore: float = previousDesigns[-1][0][1]
    
    # get the score of the first (best) design from this round
    bestCurrentScore: float = assembledScores[0][1]
    
    print('   best score is ' + str(bestCurrentScore))

    if bestCurrentScore > bestPreviousScore:
        # run another round if the scores are still improving
        
        # keep just top designs for the next round
        if len(assembledScores) > maxDesignsPerRound:
            assembledScores = assembledScores[0:maxDesignsPerRound]
        
        # recursively call self
        return designPKS(targetMol, previousDesigns=previousDesigns + [assembledScores], 
                         maxDesignsPerRound=maxDesignsPerRound, similarity=similarity)
    
    else:
        # if these designs are no better than before, just return the last round
        return previousDesigns
