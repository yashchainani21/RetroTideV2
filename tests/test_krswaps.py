"""Unit tests for the krswaps modules"""
import unittest
from parameterized import parameterized
import pytest
from unittest.mock import patch, MagicMock
from bcs import *
from krswaps import krswaps as krs
from rdkit import Chem
from cobra import Metabolite

import unittest
from unittest.mock import MagicMock, patch

class TestInputStereochemistry(unittest.TestCase):

    def test_valid_stereo_option(self):
        """Test that SMILES canonicalization works correctly."""
        smiles = 'C/C=C/[C@H]([C@H]([C@H](C(O)=O)C)O)C'
        alkene_free_smi = Chem.MolToSmiles(
            Chem.MolFromSmiles('CC=C[C@H]([C@H]([C@H](C(O)=O)C)O)C'))
        
        expected_canonical_smiles = krs.canonicalize_smiles(smiles, 'R/S')
        self.assertEqual(expected_canonical_smiles, alkene_free_smi)
    
    def test_invalid_stereo_option(self):
        """Test that an invalid stereo option raises a ValueError."""
        smiles = 'C/C=C/[C@H]([C@H]([C@H](C(O)=O)C)O)C'

        with self.assertRaises(ValueError) as context:
            krs.canonicalize_smiles(smiles, 'L/D')

class TestManualTEOffload(unittest.TestCase):
    
    def setUp(self):
        self.nonthioester_product = Chem.MolFromSmiles('C/C=C/[C@H]([C@H]([C@H](C(O)=O)C)O)C')
        self.bound_product = Chem.MolFromSmiles('C/C=C/[C@@H](C)[C@@H](O)[C@@H](C)C(S)=O')

    @parameterized.expand(['thiolysis', 'cyclization'])
    def test_missing_thioester(self, offload_mech):
        """Raises a ValueError if no thioester is found."""
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.nonthioester_product, self.nonthioester_product, offload_mech)
    
    def test_invalid_release_mechanism(self):
        """Test that an invalid release mechanism raises a ValueError."""
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.bound_product, self.bound_product, 'break_thioester_bond')
    
    def test_missing_lactone_in_target(self):
        """Test that ValueError is raised when no valid lactone is found in target molecule."""
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.bound_product, self.bound_product, 'cyclization')

    def test_no_hydroxyl_at_target_distance(self):
        """Test that ValueError is raised when no hydroxyl group is found at the
            target distance for cyclization."""
        target_molecule = Chem.MolFromSmiles('C1OCCCCCC1')
        with self.assertRaises(ValueError) as context:
            krs.te_offload(self.bound_product, target_molecule, 'cyclization')
    
    def test_successful_thiolysis(self):
        offloaded_product = krs.te_offload(self.bound_product, self.bound_product, 'thiolysis')
        self.assertEqual(Chem.MolToSmiles(offloaded_product[0]),
                         Chem.MolToSmiles(self.nonthioester_product))
    
    def test_successful_cyclization(self):
        target_molecule = Chem.MolFromSmiles('C/C=C/[C@@H](C)[C@@H](O1)[C@@H](C)C1=O')
        offloaded_product = krs.te_offload(self.bound_product, target_molecule, 'cyclization')
        self.assertEqual(Chem.MolToSmiles(offloaded_product[0]),
                         Chem.MolToSmiles(target_molecule))

class TestGetStereoMismatchResults(unittest.TestCase):

    def setUp(self):
        self.pks_product = Chem.MolFromSmiles(
            'C[C@H]1C[C@H](C)C[C@@H](C)[C@@H](O)/C(C#N)=C\C=C\C[C@@H]([C@@H]2CCC[C@H]2C(O)=O)OC(C[C@@H](O)[C@@H](C)C1)=O')
        self.target = Chem.MolFromSmiles(
            'C[C@H]1C[C@H](C[C@@H]([C@H](/C(=C\C=C\C[C@H](OC(=O)C[C@@H]([C@H](C1)C)O)[C@@H]2CCC[C@H]2C(=O)O)/C#N)O)C)C') # Borrelidin
        
        
if __name__ == '__main__':
    unittest.main()