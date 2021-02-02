#!/usr/bin/env python

# import rdkit
# from rdkit.Chem.Descriptors import MolWt
# from rdkit.Chem import Descriptors

import CGRtools
    
class FeatureExtractor():
        
    def __init__(self, df):
        self.df = df
        self.extract_features()
             
    def _get_mol_object(self):
        self.df['MOL'] = self.df.SMILES.apply(lambda x: rdkit.Chem.MolFromSmiles(x))
        
    def _add_h_atoms(self):
        self.df['MOL'] = self.df.MOL.apply(lambda x: rdkit.Chem.AddHs(x))
    
    def get_ring_info(self, x):
        ri = x.GetRingInfo()
        return ri.AtomRings()

    def _calculate_rings(self):
        self.df['num_of_rings'] = self.df.MOL.apply(lambda x: len(self.get_ring_info(x)))
        
    def _get_num_of_atoms(self):
        self.df['num_of_atoms'] = self.df.MOL.apply(lambda x: x.GetNumAtoms())
    
    def _get_num_of_heavy_atoms(self):
        self.df['num_of_heavy_atoms'] = self.df.MOL.apply(lambda x: x.GetNumHeavyAtoms())

    def _calculate_lipophilicity(self):
        self.df['logP'] = self.df.MOL.apply(lambda x: rdkit.Chem.Crippen.MolLogP(x))
        
    def _len_smiles(self):
        self.df['len_smiles'] = self.df.SMILES.apply(lambda x: len(list(x)))
        
    def _number_of_atoms(self):
        atom_list = ['C','O', 'N', 'Cl', 'Br', 'F', 'N', 'S']
        for atom in atom_list:
            self.df['num_of_{}_atoms'.format(atom)] = self.df.MOL.apply(lambda x: len(x.GetSubstructMatches(rdkit.Chem.MolFromSmiles(atom))))

    def _add_descriptors_features(self):
        self.df['tpsa'] = self.df.MOL.apply(lambda x: Descriptors.TPSA(x))
        self.df['mol_w'] = self.df.MOL.apply(lambda x: Descriptors.ExactMolWt(x))
        self.df['num_valence_electrons'] = self.df.MOL.apply(lambda x: Descriptors.NumValenceElectrons(x))
        self.df['num_heteroatoms'] = self.df.MOL.apply(lambda x: Descriptors.NumHeteroatoms(x))
        
    def drop_for_train(self):
        self.df = self.df.drop(['SMILES', 'MOL', 'SENTENCE'], axis=1)
        
    def drop_for_test(self):
        self.df = self.df.drop(['SMILES', 'MOL'], axis=1)
        
    def return_data(self):
        return self.df
    
    def extract_features(self):
        self._get_mol_object()
        self._add_h_atoms()
        self._calculate_rings()
        self._get_num_of_atoms()
        self._get_num_of_heavy_atoms()
        self._calculate_lipophilicity()
        self._len_smiles()
        self._number_of_atoms()
        self._add_descriptors_features()
        
class FeatureExtractorCGR():

      def __init__(self, smiles):
        self.df = df
        self.extract_features()

    def extract_features(self):
        reader = CGRtools.files.SMILESRead(io.StringIO(None))
        smiles = str(self.smiles)
        mol = reader.parse(smiles)
        rings = mol.aromatic_rings
        print(len(rings))