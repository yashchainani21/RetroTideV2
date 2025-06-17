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

@author: Tyler Backman, Vincent Blay
"""

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.rdchem import Mol
import itertools
from typing import Union, Callable, List, Optional, Any, Tuple

from .extras import allStarterTypes, allModuleTypes, structureDB
from bcs import Cluster, Module
from .AtomAtomPathSimilarity import getpathintegers, AtomAtomPathSimilarity
from mapchiral.mapchiral import encode, jaccard_similarity


def run_pks_release_reaction(pks_release_mechanism: str,
                             bound_product_mol: Chem.Mol) -> Optional[Chem.Mol]
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
        radius (int): 
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

        
def compareToTarget(structure: Mol,
                    target: Mol,
                    similarity: Union[str, Callable] = 'atompairs',
                    targetpathintegers=None) -> float:
    """
    Compares a given structure to a target molecule to determine their similarity.

    This function supports multiple similarity metrics, including atom pairs, maximum common substructure (MCS),
    and custom callable metrics.

    Args:
        structure (rdchem.Mol): The molecule to compare against the target.
        target (rdchem.Mol): The target molecule for comparison.
        similarity (Union[str, Callable], optional): The type of similarity metric to use. Can be 'mcs' for maximum
            common substructure, 'atompairs' for atom pair similarity, 'atomatompath' for atom-atom path similarity,
            'MAPC' for Min-hashed Atom Pair Chiral Fingerprints, 
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
    
    if similarity=='mcs':
        # MCS
        result=rdFMCS.FindMCS([target, testProduct], timeout=1, matchValences=True, matchChiralTag=True, 
                              bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact) # search for 1 second max

        if result.canceled:
            print('MCS timeout')
        score = result.numAtoms / (testProduct.GetNumAtoms() + target.GetNumAtoms() - result.numAtoms)
    
    elif similarity=='mcs_without_stereo':
        # same as above but without matching stereochemistry
        result=rdFMCS.FindMCS([target, testProduct], timeout=1, matchValences=True, matchChiralTag=False,
                              bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact) # search for 1 second max

        if result.canceled:
            print('MCS timeout')
        score = result.numAtoms / (len(testProduct.GetAtoms()) + len(target.GetAtoms()) - result.numAtoms)

    elif similarity=='atompairs':
        # atom pair
        ms = [target, testProduct]
        pairFps = [Pairs.GetAtomPairFingerprint(x) for x in ms]
        score = DataStructs.TanimotoSimilarity(pairFps[0],pairFps[1]) # can also try DiceSimilarity
        
    elif similarity=='atomatompath':
        score = AtomAtomPathSimilarity(target, testProduct, m1pathintegers=targetpathintegers)

    elif similarity=='MAPC': # Min-hashed Atom Pair Chiral Fingerprints (https://github.com/reymond-group/mapchiral)

        # compute chiral fingerprints of target and testProduct then get their jaccard similarity
        target_fp = encode(target, max_radius=2, n_permutations=2048, mapping=False)
        testProduct_fp = encode(testProduct, max_radius=2, n_permutations=2048, mapping=False)
        score = jaccard_similarity(target_fp, testProduct_fp)

    elif callable(similarity):
        score = similarity(target, testProduct)
    
    else:
        raise IOError('Invalid similarity input')
    
    return score

def designPKS(targetMol: Mol,
              previousDesigns: Optional[List[Tuple[Cluster, float, Mol]]] = None, 
              maxDesignsPerRound: int = 25,
              similarity: str = 'atompairs') -> List[List[Any]]:
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
            Supported values are 'atompairs' and 'atomatompath'. Defaults to 'atompairs'.

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
    scores: List[float] = [compareToTarget(structure, targetMol, similarity, targetpathintegers) for structure in structures]
    
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
