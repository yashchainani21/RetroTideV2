"""Unit tests for the krswaps module in stereopostprocessing."""
# pylint: disable=no-member
import unittest
from stereopostprocessing import krswaps as ks
from rdkit import Chem
import pandas as pd

test_pairs = [((1, 'LM'), (4, 'M1')),
 ((6, 'LM'), (9, 'M1')),
 ((11, 'M1'), (12, 'M2')),
 ((13, 'M2'), (14, 'M3')),
 ((15, 'M3'), (17, 'M4')),
 ((19, 'M4'), (21, 'M5')),
 ((22, 'M5'), (24, 'M6')),
 ((25, 'M6'), (27, 'M7')),
 ((28, 'M7'), (30, 'M8'))]

test_sequential_pairs = [(1, 'LM', 4, 'M1'),
 (6, 'LM', 9, 'M1'),
 (11, 'M1', 12, 'M2'),
 (13, 'M2', 14, 'M3'),
 (15, 'M3', 17, 'M4'),
 (19, 'M4', 21, 'M5'),
 (22, 'M5', 24, 'M6'),
 (25, 'M6', 27, 'M7'),
 (28, 'M7', 30, 'M8')]

test_mmatch = [17]
test_mmatch_pairs = [(15, 'M3', 17, 'M4')]

class TestIdentifyPairs(unittest.TestCase):
    """Unit tests for identifying pairs of backbone carbons."""
    def setUp(self):
        """Set up a sample unbound product for testing."""
        self.unbound_product = Chem.MolFromSmiles(
            'CC1=CC=CC[C@@H]([C@@H]2CCC[C@H]2C(=O)O)OC(=O)C[C@H](O)[C@@H](C)C[C@@H](C)C[C@@H](C)C[C@@H](C)[C@H]1O')

    def test_adjacent_carbons(self):
        """Test that adjacent backbone carbon pairs are identified correctly."""
        full_mapping_df = pd.read_csv(
            '/home/kroberts/PythonProject/Coding Practice/fully_mapped_molecule.csv')
        result = ks.find_adjacent_backbone_carbon_pairs(self.unbound_product, full_mapping_df)
        self.assertIsInstance(result, list)

    def test_sequential_pairs(self):
        """Test that sequential module pairs are filtered correctly."""
        result = ks.filter_sequential_module_pairs(test_pairs)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_mmatch_identification(self):
        """Test that pairs with chiral mismatches are reported correctly."""
        result = ks.report_pairs_with_chiral_mismatches(test_sequential_pairs, test_mmatch)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_substituent_patterns(self):
        """Test that substituent patterns are checked correctly."""
        result = ks.check_substituent_patterns(self.unbound_product, test_mmatch_pairs)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
