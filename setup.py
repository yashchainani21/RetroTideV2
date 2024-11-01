#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='retrotide',
    version='0.1',
    description='Biosynthetic Cluster Simulator',
    author='The Quantitative Metabolic Modeling group',
    author_email='tbackman@lbl.gov',
    url='https://github.com/JBEI/BiosyntheticClusterSimulator',
    packages=find_packages(),
    install_requires=[
	'cobra', 
	'numpy >= 1.8.0',
	'rdkit',
	'typing_extensions',
	'setuptools',
	'scipy'
	],
    package_data={'bcs': ['data/*'], 'retrotide': []}, 
    license='see license.txt file',
    keywords = ['biochemistry', 'synthetic biology'],
    classifiers = [],
    python_requires='>=3.6',
    )
