import spacy
from spacy.tokens import Doc, Token
from spacy.language import Language
from spacy.util import load_config
from spacy.lang.en import English

import os
from pathlib import Path
import subprocess

from numpy import mean
from typing import Callable


class KMerTokenizer:
    """
    Custom tokenizer for DNA sequences that tokenizes input DNA into k-mers.

    Attributes:
        name (str): The name of the tokenizer.
        vocab (spacy.vocab.Vocab, optional): The vocabulary object used to link words and their attributes.
        kmer_size (int, optional): The size of the k-mers to be used for tokenizing the DNA sequences.
        window_size (int, optional): The size of the window for segmenting the tokenized DNA into fixed-length sentences.
        local_model (str, optional): Path to a local pre-trained model to be used for the transformer. If None, the model will be loaded from a predefined URL.
        hf_model_path (str, optional): Path to another HuggingFace DNA model.

    Methods:
        download_model(): Downloads the transformer model if not already present.
        load_model(): Loads the transformer model either from the local model or the predefined URL.
        __call__(dna: str) -> spacy.tokens.Doc: Tokenizes the input DNA sequence into k-mers and returns a SpaCy Doc object with the tokenized sequence.

    Example:
        kmer_tokenizer = KMerTokenizer(vocab=nlp.vocab, kmer_size=5)
        dna_sequence = "ATGGCC"
        doc = kmer_tokenizer(dna_sequence)
        # doc now contains the tokenized DNA sequence
    """
    name = "kmer_tokenizer"
    def __init__(self, vocab=None, kmer_size=None, window_size=None, local_model=None, hf_model_path=None):
        self.vocab = vocab
        self.kmer_size = kmer_size
        self.window_size = window_size
        self.untokenized = ""

        if local_model:
            self.transformer = spacy.load(local_model)
        else:
            if hf_model_path == None:
                self.model_path = f"wjbmattingly/dnaBERT-k{self.kmer_size:02}-w{self.window_size:02}"
            else:
                self.model_path = hf_model_path
            self.transformer = self.load_model()

    def load_model(self):
        available_models = '\n'.join([
            "wjbmattingly/dnaBERT-k07-w10"
        ])
        # Determine the directory containing the dna_spacy file.
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Construct the path to the config file relative to dna_spacy's location.
        config_path = base_dir / "configs" / "trf_config_base.cfg"
        config = load_config(config_path)
        config['components']['transformer']['model']['name'] = self.model_path
        trf_nlp = English.from_config(config)
        try:
            trf_nlp.initialize()
        except:
            OSError
            print(f"Model not available. Please select from the following list:\n{available_models}")

        return trf_nlp

    def __call__(self, dna):
        self.untokenized = dna
        kmers = [dna[0:0+self.kmer_size]]
        end = len(dna) - self.kmer_size + 1
        for i in range(1, end): 
            kmers.append(dna[i:i+self.kmer_size])

        spaces = [True] * len(kmers)  # There are no spaces between k-mers.

        # Create the Doc object
        doc = Doc(self.vocab, words=kmers, spaces=spaces)

        # store the transformer in the doc
        doc._.transformer = self.transformer
        doc._.untokenized = self.untokenized
        return doc

@Language.factory("fixed_length_sentencizer")
def create_fixed_length_sentencizer(nlp: Language, name: str) -> Callable[[Doc], Doc]:
    """
    SpaCy pipeline component factory to create a function that segments a tokenized DNA sequence into fixed-length windows or sentences.

    This factory function returns a component that takes the tokenized DNA sequence from the Doc object and segments it into fixed-length windows based on the `window_size` custom attribute. 
    The resulting windows are stored in the custom attribute `windows` of the Doc object.

    Parameters:
        nlp (spacy.language.Language): The current nlp object, containing the language data and pipeline.
        name (str): The unique name for this component instance.

    Returns:
        Callable[[Doc], Doc]: The function that takes a SpaCy Doc object and returns it after segmenting it into fixed-length windows.

    Example:
        # Assuming the pipeline has tokenized the DNA sequence
        doc = nlp("ATGGCC")
        fixed_length_sentencizer = create_fixed_length_sentencizer(nlp, "fixed_length_sentencizer")
        fixed_length_sentencizer(doc)
        # doc._.windows now contains the fixed-length windows of the DNA sequence
    """

    def fixed_length_sentencizer(doc: Doc) -> Doc:
        window_size = doc._.get('window_size')
        windows = []
        doc_length = len(doc)
        for i in range(0, doc_length-window_size+1, window_size):
            windows.append(doc[i:i+window_size])
        if len(doc) // window_size != 0:
            windows.append(doc[-window_size:])
        doc._.windows = windows
        if len(windows) == 0:
            print("Window Size is too small.")
        return doc

    return fixed_length_sentencizer


@Language.factory("windows2vec")
def create_windows2vec(nlp: Language, name: str) -> Callable[[Doc], Doc]:
    """
    SpaCy pipeline component factory to create a function that applies a transformer to the windows of a tokenized DNA sequence.

    This factory function returns a component that takes the windows from the Doc's custom attributes and applies the stored transformer to each window,
    obtaining vectors and categories. The resulting vectors and categories are stored in the custom attributes `window_vectors` and `window_cats` of the Doc object.

    Parameters:
        nlp (spacy.language.Language): The current nlp object, containing the language data and pipeline.
        name (str): The unique name for this component instance.

    Returns:
        Callable[[Doc], Doc]: The function that takes a SpaCy Doc object and returns it after applying the transformer to the windows.

    Example:
        # Assuming the pipeline has already processed the windows
        doc = nlp("ATGGCC")
        windows2vec = create_windows2vec(nlp, "windows2vec")
        windows2vec(doc)
        # doc._.window_vectors and doc._.window_cats now contain the transformed values
    """

    def windows2vec(doc: Doc) -> Doc:
        vectors = []
        cats = []
        for window in doc._.windows:
            trf_doc = doc._.transformer(window.text)
            vectors.append(trf_doc._.trf_data.tensors[1])
            cats.append(trf_doc.cats)
        doc._.window_vectors = vectors
        doc._.window_cats = cats
        return doc

    return windows2vec


@Language.factory("average_windows")
def create_average_windows(nlp: Language, name: str) -> Callable[[Doc], Doc]:
    """
    SpaCy pipeline component factory to create a function that averages the window vectors and categories within a DNA sequence.

    This component takes the window vectors and categories (e.g., 'ANIMAL', 'BACTERIA') from the Doc's custom attributes and computes their averages. 
    The resulting averages are then stored in the `vector` and `cats` attributes of the Doc object.

    Parameters:
        nlp (spacy.language.Language): The current nlp object, containing the language data and pipeline.
        name (str): The unique name for this component instance.

    Returns:
        Callable[[Doc], Doc]: The function that takes a SpaCy Doc object and returns it after computing and setting the average vectors and categories.

    Example:
        # Assuming the pipeline has already processed window vectors and categories
        doc = nlp("ATGGCC")
        average_windows(doc)
        # doc.vector and doc.cats now contain the averaged values
    """
    def average_windows(doc: Doc) -> Doc:
        doc.vector = mean(doc._.window_vectors, axis=0)
        # Initialize a dictionary to store the sum of each category
        category_sums = {label: 0 for label in doc._.window_cats[0]}
        # Initialize a dictionary to store the count of each category
        category_counts = {label: 0 for label in doc._.window_cats[0]}

        # Iterate through the window categories and sum up the values for each label
        for item in doc._.window_cats:
            for label, value in item.items():
                category_sums[label] += value
                category_counts[label] += 1

        # Compute the average for each label
        averages = {label: total / count for label, (total, count) in zip(category_sums.keys(), zip(category_sums.values(), category_counts.values()))}

        # Assign the averages to the doc's categories
        doc.cats = averages
        return doc
    return average_windows



def DNA(kmer_size=7, window_size=10, local_model=None, hf_model_path=None):
    """
    Creates a custom SpaCy NLP pipeline to process DNA sequences by dividing them into k-mers and applying various transformations.

    Parameters:
        kmer_size (int, optional): The size of the k-mer (substring of length k) to be used for tokenizing the DNA sequences. Default is 7.
        window_size (int, optional): The size of the window for segmenting the tokenized DNA into fixed-length sentences. Default is 10.
        local_model (str, optional): Path to a local pre-trained model to be used for the transformer. If None, the model will be loaded from a predefined URL.

    Returns:
        nlp (spacy.language.Language): A SpaCy Language object configured with the custom tokenizer and pipeline for processing DNA sequences.

    The resulting pipeline includes:
        1. KMerTokenizer: Custom tokenizer that breaks the DNA into k-mers.
        2. Fixed-length Sentencizer: Divides the tokenized DNA into fixed-length windows.
        3. Windows2Vec: Applies a transformer to the windows to obtain vectors and categories.
        4. Average Windows: Averages the vectors and categories across the windows.

    Custom extensions are also added to the SpaCy Doc object to store intermediate results like untokenized DNA, window size, windows, transformer, window vectors, and window categories.

    Example:
        nlp = DNA(kmer_size=5, window_size=20)
        doc = nlp("GGCCAGGGGGCCGTTGTCCTCGGGGAACTGGCGGGCGCGCAGGTCGATCACGTCGCCGATGCGCTTGACCGCGGCCGAGCGCATGTCGGAGGTGAACTGGCTGTCGCGGAAGACGGTCAGCCCCTCCTTGAGGCAGAGCTGGAACCAATC")
        # Processed DNA sequence with custom transformations
    """
    nlp = spacy.blank("en")
    
    # Add custom attributes
    Doc.set_extension('untokenized', default=window_size, force=True)
    Doc.set_extension('window_size', default=window_size, force=True)
    Doc.set_extension('windows', default={}, force=True)
    Doc.set_extension('transformer', default=None, force=True)

    Doc.set_extension("window_vectors", default=None, force=True)
    Doc.set_extension("window_cats", default=None, force=True)


    nlp.tokenizer = KMerTokenizer(nlp.vocab, kmer_size, window_size, local_model, hf_model_path)

    nlp.add_pipe("fixed_length_sentencizer", first=True)
    nlp.add_pipe("windows2vec")
    nlp.add_pipe("average_windows")


    return nlp
