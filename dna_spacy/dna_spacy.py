import spacy
from spacy.tokens import Doc, Token
from spacy.language import Language

import torch
import torch.nn as nn
import numpy as np
import os

import srsly
import requests
import numpy as np
from numpy import mean

from . import utils

class KMerTokenizer:
    name = "kmer_tokenizer"

    def __init__(self, vocab=None, kmer_size=None, window_size=None):
        self.vocab = vocab
        self.kmer_size = kmer_size
        self.window_size = window_size
        self.letter_dict = {"A": 1, "C": 2, "G": 3, "T": 4}
        self.model_path = f"kmer_{self.kmer_size:02}_window_{self.window_size:02}.pth"
        if not os.path.exists(self.model_path):
            self.download_model(self.model_path)
        self.transformer, self.device = self.load_model(self.model_path)
        self.untokenized = ""

    def download_model(self, model_path):
        print(f"Downloading: {model_path}. This may take a few moments.")
        # Define the URL of the file
        url = f"https://huggingface.co/wjbmattingly/dna-transformer-kmer-07-window-14/resolve/main/kmer_{self.kmer_size:02}_window_{self.window_size:02}"

        # Send a GET request to the URL
        response = requests.get(url)

        # Make sure the request was successful
        response.raise_for_status()

        # Write the content of the response to a file
        with open(model_path, "wb") as f:
            f.write(response.content)

    def load_model(self, path_to_saved_model):
        # Load the saved state
        checkpoint = torch.load(path_to_saved_model)
        
        # Set the checkpoint values as attributes of self
        self.src_vocab_size = checkpoint['src_vocab_size']
        self.tgt_vocab_size = checkpoint['tgt_vocab_size']
        self.d_model = checkpoint['d_model']
        self.num_heads = checkpoint['num_heads']
        self.num_layers = checkpoint['num_layers']
        self.d_ff = checkpoint['d_ff']
        self.max_seq_length = checkpoint['max_seq_length']
        self.dropout = checkpoint['dropout']

        # Initialize the model
        model = utils.Transformer(
            src_vocab_size=checkpoint['src_vocab_size'],
            tgt_vocab_size=checkpoint['tgt_vocab_size'],
            d_model=checkpoint['d_model'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            d_ff=checkpoint['d_ff'],
            max_seq_length=checkpoint['max_seq_length'],
            dropout=checkpoint['dropout']
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode

        return model, device

    def __call__(self, dna):
        dna = dna.replace("N", '')
        self.untokenized = dna
        kmers = [dna[0:0+self.kmer_size]]
        end = len(dna) - self.kmer_size + 1
        kmer_values = np.zeros(end, dtype=np.int64)

        # Compute the initial k-mer
        kmer_value = np.int64(0)
        for i in range(self.kmer_size):
            kmer_value = kmer_value * np.int64(4) + np.int64(self.letter_dict[dna[i]])

        kmer_values[0] = kmer_value

        for i in range(1, end): 
            # Shift the previous kmer_value left by two (equivalent to multiplying by 4), 
            # and add the value of the new letter at the end
            kmer_value = kmer_value * np.int64(4) - np.int64(self.letter_dict[dna[i-1]]) * np.int64(4) ** self.kmer_size + np.int64(self.letter_dict[dna[i+self.kmer_size-1]])
            kmer_values[i] = kmer_value
            kmers.append(dna[i:i+self.kmer_size])

        spaces = [True] * len(kmers)  # There are no spaces between k-mers.

        # Create the Doc object
        doc = Doc(self.vocab, words=kmers, spaces=spaces)

        for idx, token in enumerate(doc):
            token._.numerical_value = kmer_values[idx]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # store the transformer in the doc
        doc._.transformer = self.transformer
        doc._.transformer.device = device
        doc._.untokenized = self.untokenized
        return doc

    def _get_config(self):
        return {
            'kmer_size': self.kmer_size,
            'window_size': self.window_size,
        }

    def _set_config(self, cfg):
        self.kmer_size = cfg['kmer_size']
        self.window_size = cfg['window_size']
    import os

    def to_disk(self, path, **kwargs):
        data = {
            'vocab': self.vocab.to_bytes()
        }
        srsly.write_msgpack(path, data)
        
        # Get the directory of the tokenizer file
        dir_path = os.path.dirname(path)
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        transformer_file_path = os.path.join(dir_path, "kmer_07_window_14_epoch_01.pth")
            
        torch.save({
                'model_state_dict': self.transformer.state_dict(),
                'src_vocab_size': self.src_vocab_size,
                'tgt_vocab_size': self.tgt_vocab_size,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'max_seq_length': self.max_seq_length,
                'dropout': self.dropout,
            }, transformer_file_path)
        tokenizer_path = os.path.join(dir_path, "tokenizer")
        # self.nlp.tokenizer.to_disk(tokenizer_path)


    def from_disk(self, path, **kwargs):
        data = srsly.read_msgpack(path)
        self.vocab = self.vocab.from_bytes(data['vocab'])

        # Get the directory of the tokenizer file
        dir_path = os.path.dirname(path)
        transformer_file_path = os.path.join(dir_path, "kmer_{self.kmer_size:02}_window_{self.window_size:02}.pth")

        # Load the transformer model
        checkpoint = torch.load(transformer_file_path)
        self.transformer.load_state_dict(checkpoint['model_state_dict'])
        
        # Load the other model parameters
        self.src_vocab_size = checkpoint['src_vocab_size']
        self.tgt_vocab_size = checkpoint['tgt_vocab_size']
        self.d_model = checkpoint['d_model']
        self.num_heads = checkpoint['num_heads']
        self.num_layers = checkpoint['num_layers']
        self.d_ff = checkpoint['d_ff']
        self.max_seq_length = checkpoint['max_seq_length']
        self.dropout = checkpoint['dropout']
        tokenizer_path = os.path.join(dir_path, "tokenizer")
        # self.nlp.tokenizer.from_disk(tokenizer_path)

        return self




from typing import Callable

@Language.factory("fixed_length_sentencizer")
def create_fixed_length_sentencizer(nlp: Language, name: str) -> Callable[[Doc], Doc]:
    def fixed_length_sentencizer(doc: Doc) -> Doc:
        window_size = doc._.get('window_size')
        windows = []
        doc_length = len(doc)
        for i in range(0, doc_length-window_size+1, window_size):
            windows.append(doc[i:i+window_size])
        if len(doc)//window_size != 0:
            windows.append(doc[-window_size:])
        doc._.windows = windows
        if len(windows) == 0:
            print("Window Size is too small.")

        windows_int = []
        for window in windows:
            windows_int.append(np.array([w._.numerical_value for w in window]))
        doc._.windows_int = windows_int
        return doc
    return fixed_length_sentencizer

@Language.factory("windows2vec")
def create_windows2vec(nlp: Language, name: str) -> Callable[[Doc], Doc]:
    def windows2vec(doc: Doc) -> Doc:
        vectors = []
        for window in doc._.windows:
            # Convert window to a tensor of token numerical values
            window_values = torch.tensor([token._.numerical_value for token in window]).unsqueeze(0).to(doc._.device)
            # Encode the window tensor
            vectors.append(doc._.transformer.encode(window_values)[0])
        doc._.window_vectors = vectors
        return doc
    return windows2vec

@Language.factory("average_windows")
def create_average_windows(nlp: Language, name: str) -> Callable[[Doc], Doc]:
    def average_windows(doc: Doc) -> Doc:
        doc.vector = mean(doc._.window_vectors, axis=0)
        return doc
    return average_windows

def DNA(kmer_size=7, window_size=14):
    nlp = spacy.blank("en")
    
    # Add custom attributes
    Doc.set_extension('untokenized', default=window_size, force=True)
    Doc.set_extension('window_size', default=window_size, force=True)
    Doc.set_extension('windows', default={}, force=True)
    Doc.set_extension('windows_int', default={}, force=True)
    Doc.set_extension('transformer', default=None, force=True)
    Token.set_extension("numerical_value", default=None, force=True)
    Doc.set_extension("window_vectors", default=None, force=True)
    Doc.set_extension("device", default=None, force=True)

    nlp.tokenizer = KMerTokenizer(nlp.vocab, kmer_size, window_size)

    nlp.add_pipe("fixed_length_sentencizer", first=True)
    nlp.add_pipe("windows2vec")
    nlp.add_pipe("average_windows")


    return nlp
