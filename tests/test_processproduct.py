"""Unit tests for the processproduct module in stereopostprocessing."""
# pylint: disable=no-member
import unittest
from unittest.mock import patch
from rdkit import Chem
import pandas as pd
from krswaps import processproduct as pp

TEST_SMILES = 'C[C@H]1C[C@H](C[C@@H]([C@H](/C(=C\C=C\C[C@H](OC(=O)C[C@@H]([C@H](C1)C)O)[C@@H]2CCC[C@H]2C(=O)O)/C#N)O)C)C'

class TestSmilesCanonicalization(unittest.TestCase):
    """Unit tests for SMILES canonicalization."""
    def test_e_z_removal(self):
        """Test that E/Z stereochemistry is removed correctly."""
        result = pp.canonicalize_smiles(TEST_SMILES, 'R/S')
        self.assertNotIn('/', result)
        self.assertNotIn('\\', result)

    def test_r_s_removal(self):
        """Test that R/S stereochemistry is removed correctly."""
        result = pp.canonicalize_smiles(TEST_SMILES, 'E/Z')
        self.assertNotIn('@', result)
        self.assertNotIn('@@', result)

    def test_keep_all_stereo(self):
        """Test that all stereochemistry is kept when specified."""
        result = pp.canonicalize_smiles(TEST_SMILES, 'all')
        correct_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(TEST_SMILES),
                                          isomericSmiles=True,
                                          canonical=True)
        self.assertEqual(result, correct_smiles)

    def test_all_removal(self):
        """Test that all stereochemistry is removed when specified."""
        result = pp.canonicalize_smiles(TEST_SMILES, 'none')
        correct_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(TEST_SMILES),
                                          isomericSmiles=False,
                                          canonical=True)
        self.assertEqual(result, correct_smiles)
    
    def test_invalid_stereo_type(self):
        """Test that an error is raised for invalid stereochemistry type."""
        with self.assertRaises(ValueError):
            pp.canonicalize_smiles(TEST_SMILES, 'invalid_type')

TEST_TARGET = 'C[C@@H]1C[C@H](C)C[C@H](C)[C@@H](O)CC(=O)O[C@H]([C@@H]2CCC[C@H]2C(=O)O)CC=CC=C(C#N)[C@H](O)[C@@H](C)C1'
TEST_PKS_PRODUCT = 'CC(=CC=CC[C@H](O)[C@@H]1CCC[C@H]1C(=O)O)[C@H](O)[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)[C@@H](O)CC(=O)[S]'
class TestProductOffloading(unittest.TestCase):
    """Unit tests for offloading PKS products."""
    def test_thiolysis(self):
        """Test that thiolysis offloading works correctly."""
        result = pp.offload_pks_product(Chem.MolFromSmiles(TEST_PKS_PRODUCT),
                                        Chem.MolFromSmiles(TEST_TARGET),
                                        'thiolysis')
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], Chem.Mol)
        self.assertNotIn('S', Chem.MolToSmiles(result[0]), 'Sulfur should be removed after thiolysis')

        terminal_acid_pattern = Chem.MolFromSmarts('[C](=[O])[OH]')
        acid_matches = result[0].GetSubstructMatches(terminal_acid_pattern)
        self.assertGreater(len(acid_matches), 0, 'Product should contain terminal acid group')
    
    def test_cyclization(self):
        """Test that cyclization offloading works correctly."""
        result = pp.offload_pks_product(Chem.MolFromSmiles(TEST_PKS_PRODUCT),
                                        Chem.MolFromSmiles(TEST_TARGET),
                                        'cyclization')
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], Chem.Mol)
        self.assertNotIn('S', Chem.MolToSmiles(result[0]),
                         'Sulfur should be removed after cyclization')

        ester_pattern = Chem.MolFromSmarts('[C](=[O])[O]')
        ester_matches = result[0].GetSubstructMatches(ester_pattern)
        self.assertGreater(len(ester_matches), 0,
                           'Product should contain ester group(s) after cyclization')
    
    @patch('builtins.print')
    def test_no_thioester_groups(self, mock_print):
        """Test that a warning is printed when no thioester groups are found."""
        no_thioester_product = 'CCCC'
        result = pp.offload_pks_product(Chem.MolFromSmiles(no_thioester_product),
                                        Chem.MolFromSmiles(TEST_TARGET),
                                        'cyclization')
        mock_print.assert_called_with('Warning: No thioester found in PKS product')

    @patch('builtins.print')
    def test_no_hydroxyl_groups(self, mock_print):
        """Test that a warning is printed when no hydroxyl groups are found."""
        no_oh_product = 'CCCCC(=O)S'
        result = pp.offload_pks_product(Chem.MolFromSmiles(no_oh_product),
                                        Chem.MolFromSmiles(TEST_TARGET),
                                        'cyclization')
        mock_print.assert_called_with('Warning: No hydroxyl groups found')

    @patch('builtins.print')
    def test_no_oh_at_target_distance(self, mock_print):
        """Test that a warning is printed when no hydroxyl group is found at the target distance."""
        no_oh_target_distance = 'C(O)CC(=O)S'
        result = pp.offload_pks_product(Chem.MolFromSmiles(no_oh_target_distance),
                                        Chem.MolFromSmiles(TEST_TARGET),
                                        'cyclization')
        found = any(
            'No hydroxyl found at target distance' in str(call.args[0])
            for call in mock_print.call_args_list
            if call.args
        )
        self.assertTrue(found, "Should print a failure message with distance")

TEST_UNBOUND_PRODUCT = 'CC1=CC=CC[C@@H]([C@@H]2CCC[C@H]2C(=O)O)OC(=O)C[C@H](O)[C@@H](C)C[C@@H](C)C[C@@H](C)C[C@@H](C)[C@H]1O'
class TestProductMapping(unittest.TestCase):
    """Unit tests for mapping unbound products to targets."""
    def check_mcs_match(self):
        """Test that finding structural correspondence using MCS works correctly."""
        result = pp.matching_target_atoms(Chem.MolFromSmiles(TEST_UNBOUND_PRODUCT),
                                          Chem.MolFromSmiles(TEST_TARGET))
        self.assertIsInstance(result[0], tuple)
        self.assertIsInstance(result[1], Chem.Mol)

    def test_mcs_mapping(self):
        """Test that a dataframe is returned with MCS mapping."""
        result = pp.map_product_to_target(Chem.MolFromSmiles(TEST_UNBOUND_PRODUCT),
                                              Chem.MolFromSmiles(TEST_TARGET))
        self.assertIsInstance(result, pd.DataFrame)

class TestChiralCenters(unittest.TestCase):
    """Unit tests for checking chiral centers in mapped product and target."""
    def test_chiral_centers(self):
        """Test that chiral centers matches and mismatches are identified correctly."""
        full_mapping_df = pd.read_csv('/home/kroberts/PythonProject/Coding Practice/fully_mapped_molecule.csv')
        result = pp.check_chiral_centers(Chem.MolFromSmiles(TEST_UNBOUND_PRODUCT),
                                         Chem.MolFromSmiles(TEST_TARGET),
                                         full_mapping_df)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
        self.assertIsInstance(result[2], list)
        self.assertIsInstance(result[3], list)
        self.assertIsInstance(result[4], dict)
        self.assertIsInstance(result[5], dict)

if __name__ == '__main__':
    unittest.main()
