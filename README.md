![dna spacy logo](images/dna-spacy-logo.png)

# About

DNA spaCy is a machine learning [spaCy](www.spacy.io) pipeline for processing DNA sequences found in FASTA files. By treating the classification of DNA as an NLP problem, we can leverage NLP libraries, such as spaCy to classify DNA sequences. This methodology has already been explored by other [scholars](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680). To our knowledge, however, these methods have not been applied in a spaCy pipeline. We selected spaCy because of its wide use in industry and academia, speed, and its Python library is accessible and expandable.

The initial layer of the DNA spaCy layer is a custom tokenizer designed to work with DNA sequences which are long sequences of unseparated letters that correspond to nucleotides. Traditionally, scientists divide these sequences into KMERs, or substrings of this representation. This is important for downstream tasks like classification. The KMER tokenizer allows the user to pass a custom kmer size to tokenize the text. In addition to this, the KMER Tokenizer also converts the KMER sequences into unique integers. The KMERs are then subdivided into windows of a pre-specified length. These stages are important because they are the way in which our encoding layer, a transformer, was trained.

By default, the doc container represents the tokenized text (each KMER in sequential order). In order to access the original string input, you can use `doc._.untokenized`.

The windows are provided at `doc._.windows` and their numeric representations are provided at: `doc._.windows_int = windows_int`.

The next layer in the pipeline is an embedding layer. This is handled by a transformer model that was trained on ~43 million DNA sequences from 23 marine wildlife species and a collection of bacteria reads. Each window is sent to the transformer model and encoded.

A final step averages out the encodings of each window for a given input. This is stored in `doc.vector`.