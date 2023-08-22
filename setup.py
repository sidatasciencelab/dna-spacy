from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

VERSION = '0.0.1'
DESCRIPTION = 'A spaCy library for working with DNA Sequences.'

setup(
    name="dna_spacy",
    author="Jennifer Spillane and WJB Mattingly",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'spacy==3.6.1',
        'spacy-transformers==1.2.5',
        'scikit-learn==1.3.0'
    ],
    package_data={
        'dna_spacy': ['configs/trf_config_base.cfg']
    },
    include_package_data=True
)
