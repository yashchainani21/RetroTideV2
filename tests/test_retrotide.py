import unittest
from unittest.mock import call, patch, MagicMock
from rdkit import Chem
from rdkit.Chem import rdFMCS, DataStructs
from collections import OrderedDict
from bcs import *
from rdkit.Chem.rdchem import Mol
from retrotide import *

class TestCompareToTarget(unittest.TestCase):
    def setUp(self):
        # Simple molecules for testing
        self.structure = Chem.MolFromSmiles('CC(=O)S')
        self.target = Chem.MolFromSmiles('CCO')

    def test_custom_callable_metric(self):
        custom_metric = MagicMock(return_value=0.5)
        score = compareToTarget(self.structure, self.target, custom_metric)
        self.assertEqual(score, 0.5)
        custom_metric.assert_called_once()

    def test_invalid_similarity_input(self):
        retrotide.targetpathintegers = None
        with self.assertRaises(IOError):
            compareToTarget(self.structure, self.target, 'invalid_metric')

    @patch('retrotide.retrotide.AtomAtomPathSimilarity')
    @patch('rdkit.Chem.rdmolops.ReplaceSubstructs')
    def test_atom_atom_path(self, mock_replace_substructs, mock_similarity):
        # Set the mock return value for the similarity function
        mock_similarity.return_value = 0.75
        mocked_product = MagicMock()
        mock_replace_substructs.return_value = (mocked_product,)
        mock_target_path_integers = MagicMock()

        # Call the function with 'atomatompath' to trigger the AtomAtomPathSimilarity path
        score = compareToTarget(self.structure, self.target, 'atomatompath', mock_target_path_integers)

        # Check if AtomAtomPathSimilarity was called correctly
        mock_similarity.assert_called_once_with(self.target, mocked_product, m1pathintegers=mock_target_path_integers)

        # Assert that the function returns what the mock is configured to return
        self.assertEqual(score, 0.75)

    @patch('rdkit.Chem.rdFMCS.FindMCS')
    def test_mcs_similarity(self, mock_mcs):
        structure = Chem.MolFromSmiles('CCCCCC')
        target = Chem.MolFromSmiles('SSSSSS')

        # Assuming that the compareToTarget uses numAtoms to calculate the score
        # We mock the MCS result to manipulate the score calculation
        mcs_result = MagicMock(canceled=False, numAtoms=10)
        # Mock the return value of FindMCS
        mock_mcs.return_value = mcs_result

        mcs_result.numAtoms = 10

        score = compareToTarget(structure, target, 'mcs')
        
        self.assertEqual(score, 5)
        mock_mcs.assert_called_once()

    @patch('rdkit.Chem.rdFMCS.FindMCS')
    def test_similarity_replace_substruct(self, mock_mcs):
        structure = Chem.MolFromSmiles('C(=O)S')
        target = Chem.MolFromSmiles('S')

        # Assuming that the compareToTarget uses numAtoms to calculate the score
        # We mock the MCS result to manipulate the score calculation
        mcs_result = MagicMock(canceled=False, numAtoms=10)
        # Mock the return value of FindMCS
        mock_mcs.return_value = mcs_result

        mcs_result.numAtoms = 1

        score = compareToTarget(structure, target, 'mcs')
        
        self.assertEqual(score, 1)
        mock_mcs.assert_called_once()

    @patch('rdkit.Chem.DataStructs.TanimotoSimilarity')
    @patch('rdkit.Chem.AtomPairs.Pairs.GetAtomPairFingerprint')
    @patch('rdkit.Chem.rdmolops.ReplaceSubstructs')
    def test_atompairs(self, mock_replace_substructs, mock_get_fingerprint, mock_tanimoto_similarity):
        # Mock the atom pair fingerprints to control the output
        mock_fp1 = MagicMock(spec=DataStructs.UIntSparseIntVect)
        mock_fp2 = MagicMock(spec=DataStructs.UIntSparseIntVect)
        mock_get_fingerprint.side_effect = [mock_fp1, mock_fp2]
        
        mocked_product = MagicMock()
        mock_replace_substructs.return_value = (mocked_product,)
        # Mock TanimotoSimilarity to return a controlled, expected score
        expected_score = 0.75
        mock_tanimoto_similarity.return_value = expected_score

        # Call the function with 'atompairs' to trigger the atom pair similarity path
        score = compareToTarget(self.structure, self.target, 'atompairs')

        # Ensure the fingerprints are generated for both molecules
        mock_get_fingerprint.assert_any_call(mocked_product)
        mock_get_fingerprint.assert_any_call(self.target)

        # Check that TanimotoSimilarity is called with the mocked fingerprints
        mock_tanimoto_similarity.assert_called_once_with(mock_fp1, mock_fp2)

        # Assert that the function returns the expected score
        self.assertEqual(score, expected_score)

# Limit the set of starter types and module types for the sake of testing
mock_starter_mod = MagicMock(spec=Module, name="Mock Starter Module")
mock_starter_mod.computeProduct.return_value = 'mocked_product'
mockAllStarterTypes = [mock_starter_mod]

mock_all_module_types = [MagicMock(spec=Module, name="Module Type") for _ in range(3)]  # Create 3 mock modules

@patch('retrotide.retrotide.allStarterTypes', new=mockAllStarterTypes)
@patch('retrotide.retrotide.allModuleTypes', new=mock_all_module_types)
class TestDesignPKS(unittest.TestCase):
    def setUp(self):
        from retrotide import designPKS
        self.target_mol = MagicMock()

    @patch('retrotide.retrotide.Cluster')
    @patch('retrotide.retrotide.compareToTarget')
    def test_designPKS_previous_designs_no_improvement(self,
                                              mock_compare_to_target,
                                              mock_cluster):
        """Test the designPKS function when there are previous designs and no improvement in similarity score for the round."""
        # Setup
        mock_compare_to_target.return_value = 0 # Mock the comparison function
        mocked_cluster_instance = MagicMock(spec=Cluster, name="Mocked Next Cluster")
        mock_cluster.return_value = mocked_cluster_instance  # Mock instances of Cluster
        mocked_cluster_instance.computeProduct.return_value = 'Mocked Product'

        # Prepare mock inputs for previousDesigns and allModuleTypes
        prev_cluster = MagicMock(spec=Cluster, name="Mocked Previous Cluster")
        prev_structure = MagicMock(spec=Mol)
        mock_previous_designs = [
            (prev_cluster, 0.5, prev_structure)
        ]
        mock_prev_designs_module = MagicMock(spec=Module, name="Mocked Previous Module")

        mock_prev_designs_modules = [mock_prev_designs_module]
        prev_cluster.modules = mock_prev_designs_modules
        cluster_calls = [call(modules=mock_prev_designs_modules + [mod]) for mod in mock_all_module_types]

        # Directly calling designPKS or the method that includes the list comprehension
        # Ensure designPKS is capable of accepting these mocks as parameters or adjust your function's design
        result = designPKS(targetMol=self.target_mol,
                           previousDesigns=[mock_previous_designs],
                           maxDesignsPerRound=10,
                           similarity='atompairs')

        mock_cluster.assert_has_calls(cluster_calls, any_order=True)
        calls_to_computeProduct = [call(structureDB, chain=prev_structure)]
        mocked_cluster_instance.computeProduct.assert_has_calls(calls_to_computeProduct, any_order=True)

        # Validate that the clusters were constructed with expected modules
        expected_calls_to_compare_target = [call('Mocked Product', self.target_mol, 'atompairs', None) for _ in range(len(cluster_calls))]
        mock_compare_to_target.assert_has_calls(expected_calls_to_compare_target, any_order=True)

        # Assuming designPKS returns a list of designs which include Cluster instances
        # Check the result for the correct application of mock values
        self.assertEqual(result, [mock_previous_designs]) # The similarity score decreased this round, so we should return the previous designs

    @patch('retrotide.retrotide.Cluster')
    @patch('retrotide.retrotide.compareToTarget')
    def test_designPKS_no_previous_designs_no_improvement(self,
                                           mock_compare_to_target,
                                           mock_cluster):
        """Test the designPKS function when there are no previous designs and no improvement in similarity score for the round."""
        # Arrange
        mock_compare_to_target.return_value = 0 # Mock the comparison function
        mocked_starter_cluster_instance = MagicMock(spec=Cluster, name='Starter Cluster')
        mocked_starter_cluster_instance.computeProduct.return_value = 'Mocked Starter Product'
        mocked_starter_cluster_instance.modules = mockAllStarterTypes

        mocked_next_design_cluster_instance = MagicMock(spec=Cluster, name="Mocked Next Cluster")
        mocked_next_design_cluster_instance.computeProduct.return_value = 'Mocked Next Product'

        # Mock the Cluster class to return the starter cluster instance for calls with the starter module and the next design cluster instance for the rest
        def cluster_side_effect(*args, **kwargs):
            modules = kwargs['modules']
            if modules == mockAllStarterTypes:
                return mocked_starter_cluster_instance
            return mocked_next_design_cluster_instance
        
        mock_cluster.side_effect = cluster_side_effect

        previousDesigns = []
        for _ in range(len(mockAllStarterTypes)):
            previousDesigns.append((mocked_starter_cluster_instance, 0.0, mocked_starter_cluster_instance.computeProduct()))

        prev_structures = [prev_design[-1] for prev_design in previousDesigns]
        cluster_call_starters = [call(modules=[starter_mod]) for starter_mod in mockAllStarterTypes]
        cluster_call_new_designs = [call(modules=mockAllStarterTypes + [mod]) for mod in mock_all_module_types]
        calls_to_computeProduct = [call(structureDB, chain=prev_structure) for prev_structure in prev_structures] * len(cluster_call_new_designs)
        expected_calls_to_compare_target = [call('Mocked Next Product', self.target_mol, 'atompairs', None) for _ in range(len(cluster_call_new_designs))]

        # act
        result = designPKS(targetMol=self.target_mol,
                           maxDesignsPerRound=10,
                           similarity='atompairs')
        
        # Assert
        mock_cluster.assert_has_calls(cluster_call_starters, any_order=True)
        mock_cluster.assert_has_calls(cluster_call_new_designs, any_order=True)

        mocked_next_design_cluster_instance.computeProduct.assert_has_calls(calls_to_computeProduct, any_order=True)
        mock_compare_to_target.assert_has_calls(expected_calls_to_compare_target, any_order=True)
        self.assertEqual(result, [previousDesigns]) # The similarity score decreased this round, so we should return the previous designs

    @patch('retrotide.retrotide.Cluster')
    @patch('retrotide.retrotide.compareToTarget')
    @patch('retrotide.retrotide.designPKS')
    def test_designPKS_run_another_round_when_there_was_improvement(self,
                                                          mock_designPKS,
                                                          mock_compare_to_target,
                                                          mock_cluster):
        """Test the designPKS function when there is an improvement in similarity score for the round.
        In this case, we should make a recursive call to designPKS."""
        # Arrange
        mock_designPKS.return_value = "Mocked Design Result"
        mock_compare_to_target.return_value = 1 # Mock the comparison function
        mocked_starter_cluster_instance = MagicMock(spec=Cluster, name='Starter Cluster')
        mocked_starter_cluster_instance.computeProduct.return_value = 'Mocked Starter Product'
        mocked_starter_cluster_instance.modules = mockAllStarterTypes

        mocked_next_design_cluster_instance = MagicMock(spec=Cluster, name="Mocked Next Cluster")
        mocked_next_design_cluster_instance.computeProduct.return_value = 'Mocked Next Product'

        # Mock the Cluster class to return the starter cluster instance for calls with the starter module and the next design cluster instance for the rest
        def cluster_side_effect(*args, **kwargs):
            modules = kwargs['modules']
            if modules == mockAllStarterTypes:
                return mocked_starter_cluster_instance
            return mocked_next_design_cluster_instance
        
        mock_cluster.side_effect = cluster_side_effect

        previousDesigns = []
        for _ in range(len(mockAllStarterTypes)):
            previousDesigns.append((mocked_starter_cluster_instance, 0.0, mocked_starter_cluster_instance.computeProduct()))

        assembledScores = [(mocked_next_design_cluster_instance, 1, 'Mocked Next Product') for _ in range(len(mock_all_module_types))]
        
        # act
        maxDesigns = 2
        result = designPKS(targetMol=self.target_mol,
                           maxDesignsPerRound=maxDesigns,
                           similarity='atompairs')
        
        # Assert
        # Ensure that previousDesigns is a list of lists, each representing a design round
        previousDesigns = [previousDesigns]  # This makes it a list containing one round of designs.

        # mock_designPKS.assert_has_calls(next_round_calls)
        mock_designPKS.assert_called_once_with(self.target_mol,
            previousDesigns=previousDesigns + [assembledScores[:maxDesigns]],
            maxDesignsPerRound=maxDesigns,
            similarity='atompairs')
        self.assertEqual(result, "Mocked Design Result")

if __name__ == '__main__':
    unittest.main()
