import spacy

class DNAParser:

    def __init__(self):
        self.nlp = spacy.load("dna_md")
    
    def process_doc(self, text):

        #open the fasta file as a generator

        #process each sequence
        docs = {}

        for i, sequence in enumerate(sequences):
            doc = self.nlp(sequence)
            docs[i] = {"doc": doc, "sequence": sequence}
        return docs
