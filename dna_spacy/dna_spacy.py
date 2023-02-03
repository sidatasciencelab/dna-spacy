import spacy
from spacy.tokens import Doc

class DNAParser:

    def __init__(self, model):
        self.nlp = spacy.load(model)

    def load_fasta(self, fp):
        pass
    
    def create_kmer(self, sequence, k):
        kmers = []
        end = len(sequence)-k+1
        for i, c in enumerate(sequence):
            if i < end:
                kmers.append(f"{sequence[i:i+k]}")
        return kmers

    
    def process_doc(self, sequence, k=3):
        """
        This function will process a DNA sequence, convert it to a 3-mer, and turn it into a spaCy
        doc container

        ----
        Args:
            sequence (str) => a DNA sequence that is not split into a k-mer

        Returns:
            doc (spaCy Doc container) => a doc container representation of the input sequence
            to access the doc vector, you can use doc.vector
        """
        Doc.set_extension("sequence", default=False)

        kmers = self.create_kmer(sequence, k)
        kmer_sequence = " ".join(kmers)

        # create the doc containner
        doc = self.nlp(kmer_sequence)
        doc._.sequence = sequence

        return doc
