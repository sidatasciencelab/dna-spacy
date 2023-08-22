![dna spacy logo](images/dna-spacy-logo.png)

# About

DNA spaCy is a machine learning [spaCy](www.spacy.io) pipeline for processing DNA sequences found in FASTA files. By treating the classification of DNA as an NLP problem, we can leverage NLP libraries, such as spaCy to classify DNA sequences. This methodology has already been explored by other [scholars](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680). To our knowledge, however, these methods have not been applied in a spaCy pipeline. We selected spaCy because of its wide use in industry and academia, speed, and its Python library is accessible and expandable.

The initial layer of the DNA spaCy layer is a custom tokenizer designed to work with DNA sequences represented as long sequences of unseparated letters that correspond to nucleotides. Traditionally, scientists divide these sequences into kmers, or substrings of length k. This is important for downstream tasks like classification, genome assembly, metagenomics, and many other applications. Here, the kmers correspond to "words" in the DNA sequence. The KMER tokenizer allows the user to pass a custom kmer size to tokenize the text. In addition to this, the KMER Tokenizer also converts the kmer sequences into unique integers. The KMERs are then subdivided into windows of a pre-specified length, which help alleviate the problem of DNA sequences of different lengths. These stages are important because they are the way in which our encoding layer, a transformer, was trained.

By default, the doc container represents the tokenized text (each KMER in sequential order). In order to access the original string input, you can use `doc._.untokenized`.

The windows are provided at `doc._.windows`.

The next layer in the pipeline is an embedding layer. This is handled by a transformer model that was trained on ~43 million DNA sequences from 23 marine wildlife species and a collection of bacteria reads. Each window is sent to the transformer model and encoded.

A final step averages out the encodings of each window for a given input. This is stored in `doc.vector`.

This library is meant to be used alongside DNA spaCy Project which allows for you to train downstream tasks, such as DNA classification. This can be performed on each individual window, rather than the doc as a whole, allowing for a nuanced classification of portions of a DNA sequence.

# Installation

```bash
pip install dna-spacy
```

# Usage

First, import the `DNA` class and instantiate it. Here you can specify the kmer size and window size that you want to use. Currently, the transformer models only supported are `kmer_size=7` and `window_size=10`

```python
from dna_spacy import dna_spacy as ds
nlp = ds.DNA(kmer_size=7, window_size=10)
```

Next, pass a sequence to the class and print it off.

```python
dna = "TTCGGGGTGGCCCTGCAGGCGACGGGCGCGCCTGCGGTCCTCCGTGAACCCGGTGCGGTGACGATGCGCGTGCACGACGTCGACCTGCGGGTCTCGGCCGGGGCGTTCTTCCAGGCCGGCCCGGCCGCGGCTGCGGCGCTGGTCGACCTC"

doc = nlp(dna)
print(doc)
```

```python
TTCGGGG TCGGGGT CGGGGTG GGGGTGG GGGTGGC GGTGGCC GTGGCCC TGGCCCT GGCCCTG GCCCTGC CCCTGCA CCTGCAG CTGCAGG TGCAGGC GCAGGCG
CAGGCGA AGGCGAC GGCGACG GCGACGG CGACGGG GACGGGC ACGGGCG CGGGCGC GGGCGCG GGCGCGC GCGCGCC CGCGCCT GCGCCTG CGCCTGC GCCTGCG 
CCTGCGG CTGCGGT TGCGGTC GCGGTCC CGGTCCT GGTCCTC GTCCTCC TCCTCCG CCTCCGT CTCCGTG TCCGTGA CCGTGAA CGTGAAC GTGAACC TGAACCC 
GAACCCG AACCCGG ACCCGGT CCCGGTG CCGGTGC CGGTGCG GGTGCGG GTGCGGT TGCGGTG GCGGTGA CGGTGAC GGTGACG GTGACGA TGACGAT GACGATG 
ACGATGC CGATGCG GATGCGC ATGCGCG TGCGCGT GCGCGTG CGCGTGC GCGTGCA CGTGCAC GTGCACG TGCACGA GCACGAC CACGACG ACGACGT CGACGTC 
GACGTCG ACGTCGA CGTCGAC GTCGACC TCGACCT CGACCTG GACCTGC ACCTGCG CCTGCGG CTGCGGG TGCGGGT GCGGGTC CGGGTCT GGGTCTC GGTCTCG 
GTCTCGG TCTCGGC CTCGGCC TCGGCCG CGGCCGG GGCCGGG GCCGGGG CCGGGGC CGGGGCG GGGGCGT GGGCGTT GGCGTTC GCGTTCT CGTTCTT GTTCTTC 
TTCTTCC TCTTCCA CTTCCAG TTCCAGG TCCAGGC CCAGGCC CAGGCCG AGGCCGG GGCCGGC GCCGGCC CCGGCCC CGGCCCG GGCCCGG GCCCGGC CCCGGCC 
CCGGCCG CGGCCGC GGCCGCG GCCGCGG CCGCGGC CGCGGCT GCGGCTG CGGCTGC GGCTGCG GCTGCGG CTGCGGC TGCGGCG GCGGCGC CGGCGCT GGCGCTG 
GCGCTGG CGCTGGT GCTGGTC CTGGTCG TGGTCGA GGTCGAC GTCGACC TCGACCT CGACCTC
```

You can also access the original untokenized text via the custom extension `doc._.untokenized`

```python
print(doc._.untokenized)
```

```python
TTCGGGGTGGCCCTGCAGGCGACGGGCGCGCCTGCGGTCCTCCGTGAACCCGGTGCGGTGACGATGCGCGTGCACGACGTCGACCTGCGGGTCTCGGCCGGGGCGTTCTTCCAGGCCGGCCCGGCCGCGGCTGCGGCGCTGGTCGACCTC
```

The `Doc` container also contains custom extensions that allow you to access the individual windows:

```python
for window in doc._.windows:
    print(window)
```


```python
TTCGGGG TCGGGGT CGGGGTG GGGGTGG GGGTGGC GGTGGCC GTGGCCC TGGCCCT GGCCCTG GCCCTGC
CCCTGCA CCTGCAG CTGCAGG TGCAGGC GCAGGCG CAGGCGA AGGCGAC GGCGACG GCGACGG CGACGGG
...
```

Each window has its own vector

```python
for window_vec in doc._.window_vectors:
    print(window_vec)
```

```python
[[-7.39537120e-01  6.82567656e-01  2.06507981e-01 -2.69144595e-01
  ...,
  -7.39537120e-01  6.82567656e-01  2.06507981e-01 -2.69144595e-01
  ...
]]
```

The library also averages out each window to capture the entire sequence's vector. It automatically places this in the `doc.vector` extension.


```python
print(doc.vector)
```

```python
[[-7.39537120e-01  6.82567656e-01  2.06507981e-01 -2.69144595e-01
  -5.84327400e-01 -4.37274933e-01 -9.49760228e-02  6.02423847e-01
  -2.15590253e-01 -1.82508811e-01 -2.84874998e-02 -1.85540672e-02
  -3.08907479e-01  3.78764182e-01 -3.48590463e-01  1.51973039e-01
  ...
]]
```
