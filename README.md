![dna spacy logo](images/dna-spacy-logo.png)

# About

DNA spaCy is a machine learning [spaCy](www.spacy.io) pipeline for processing DNA sequences found in FASTA files. By treating the classification of DNA as an NLP problem, we can leverage NLP libraries, such as spaCy to classify DNA sequences. This methodology has already been explored by other [scholars](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680). To our knowledge, however, these methods have not been applied in a spaCy pipeline. We selected spaCy because of its wide use in industry and academia, speed, and its Python library is accessible and expandable.

The initial layer of the DNA spaCy layer is an embedding layer that was trained on DNA sequences with a kmer of 3. For this process, we leveraged [Floret embeddings](https://github.com/explosion/floret), a spaCy extension of Bloom embeddings of the [fastText](https://fasttext.cc/). Floret embeddings allow for spaCy to leverage sub-word embeddings which keeps the embedding layer small and efficient.

The DNA spaCy's method `process_doc()` has a pre-processing step that will automatically convert your DNA sequences into a 3-mer, thus aligning your DNA sequences with the model's expected input. The pipeline then produces a vector for each DNA sequence.