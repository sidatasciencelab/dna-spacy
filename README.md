![dna spacy logo](images/dna-spacy-logo.png)

# About

DNA spaCy is a machine learning [spaCy](www.spacy.io) pipeline for processing DNA sequences found in FASTA files. By treating the classification of DNA as an NLP problem, we can leverage NLP libraries, such as spaCy to classify DNA sequences. This methodology has already been explored by other [scholars](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680). To our knowledge, however, these methods have not been applied in a spaCy pipeline. We selected spaCy because of its wide use in industry and academia, speed, and its Python library is accessible and expandable.

The initial layer of the DNA spaCy layer is a custom tokenizer designed to work with DNA sequences which are long sequences of unseparated letters that correspond to nucleotides. Traditionally, scientists divide these sequences into KMERs, or substrings of this representation. This is important for downstream tasks like classification. The KMER tokenizer allows the user to pass a custom kmer size to tokenize the text. In addition to this, the KMER Tokenizer also converts the KMER sequences into unique integers. The KMERs are then subdivided into windows of a pre-specified length. These stages are important because they are the way in which our encoding layer, a transformer, was trained.

By default, the doc container represents the tokenized text (each KMER in sequential order). In order to access the original string input, you can use `doc._.untokenized`.

The windows are provided at `doc._.windows` and their numeric representations are provided at: `doc._.windows_int = windows_int`.

The next layer in the pipeline is an embedding layer. This is handled by a transformer model that was trained on ~43 million DNA sequences from 23 marine wildlife species and a collection of bacteria reads. Each window is sent to the transformer model and encoded.

A final step averages out the encodings of each window for a given input. This is stored in `doc.vector`.

# Installation

1. Clone repository
```bash
git clone https://github.com/sidatasciencelab/dna-spacy
```

2. Move into the dna-spacy directory

```bash
cd dna-spacy
```

3. Install required libraries

```bash
pip install -r requirements.txt
```

# Usage

```python
from dna_spacy import dna_spacy as ds
nlp = ds.DNA()
doc = nlp("ACGGTCATCATAAAGGCAGTGATCGCACCCTCTGACAGACGTCATGTTAATA")
print("Tokenized DNA:")
print(doc)
print("Original DNA:")
print(doc._.untokenized)
print("DNA Vector:")
print(doc.vector)
```

## Expected Output:
```python
Tokenized DNA:
CGGTCAT GGTCATC GTCATCA TCATCAT CATCATA ATCATAA TCATAAA CATAAAG ATAAAGG TAAAGGC AAAGGCA AAGGCAG AGGCAGT GGCAGTG GCAGTGA CAGTGAT AGTGATC GTGATCG TGATCGC GATCGCA ATCGCAC TCGCACC CGCACCC GCACCCT CACCCTC ACCCTCT CCCTCTG CCTCTGA CTCTGAC TCTGACA CTGACAG TGACAGA GACAGAC ACAGACG CAGACGT AGACGTC GACGTCA ACGTCAT CGTCATG GTCATGT TCATGTT CATGTTA ATGTTAA TGTTAAT GTTAATA 
Original DNA:
ACGGTCATCATAAAGGCAGTGATCGCACCCTCTGACAGACGTCATGTTAATA
DNA Vector:
[-1.58193022e-01 -2.42662638e-01  1.90813601e-01 -7.27092147e-01
  1.55217975e-01 -5.85851073e-01  1.73725009e-01 -3.77077796e-03
  7.06247017e-02 -1.04803927e-01  3.64086151e-01 -1.41671985e-01
  3.66305441e-01  3.63368914e-02 -5.35046458e-02 -2.79592331e-02
 -5.82682453e-02  9.53428000e-02 -2.83256769e-01  4.14657295e-01
  1.20611988e-01 -2.24542156e-01 -4.51667309e-02  9.87707227e-02
 -1.67543620e-01 -1.41439795e-01 -2.43126586e-01  1.08223110e-01
  7.11639374e-02  5.45557588e-02 -1.46427289e-01 -1.63550675e-01
 -7.94471502e-02 -3.10851395e-01  2.06379235e-01 -3.39359015e-01
 -3.41869593e-01  4.41905409e-02  3.71017605e-02 -6.05896115e-02
 -5.78298494e-02 -2.04302758e-01 -2.76452363e-01  9.53331217e-03
 -6.34741485e-02 -4.93553728e-02 -1.03615131e-03  1.15189455e-01
  1.50297716e-01  1.67644508e-02 -1.61373734e-01 -2.69836098e-01
  6.19084463e-02 -3.14643309e-02  3.45466696e-02  1.83165818e-01
  1.59629434e-01  3.05753559e-01 -5.76136634e-02 -2.28798449e-01
 -3.01358057e-03  2.76046991e-01  6.07232928e-01 -1.55448735e-01
  1.75184593e-01  3.47386956e-01 -1.15964845e-01  6.10891804e-02
  2.35830620e-03  1.88163459e-01 -1.34398732e-02  5.33045381e-02
  1.81047261e-01  4.67149280e-02  3.86074781e-01  1.30868599e-01
  2.95731723e-02 -6.70329761e-03 -3.39480579e-01 -9.84811559e-02
 -2.07793161e-01  6.05944879e-02  1.81168377e-01 -1.09077595e-01
 -1.49289653e-01  3.72099608e-01  1.59347236e-01 -2.68875480e-01
 -1.85315281e-01  1.04586102e-01 -3.38473588e-01  8.79495591e-03
  3.46449345e-01  9.52551365e-02  2.20317021e-01 -4.62998487e-02
 -3.06254536e-01  1.25159353e-01  1.58919543e-01 -5.73351830e-02
  1.09668821e-01 -2.98845991e-02 -2.57855177e-01  1.35276750e-01
 -2.29545057e-01 -9.51001607e-03  1.11466020e-01  7.67843500e-02
  6.27833903e-02 -3.98864597e-02  5.51060773e-04 -1.49372190e-01
  5.60535789e-02  3.40665042e-01 -1.25835568e-01 -2.36775041e-01
 -4.63149920e-02  1.75175130e-01 -2.14395866e-01  1.46581411e-01
  2.46373832e-01  1.57109544e-01 -2.57251650e-01  5.30188456e-02
 -2.46004075e-01  9.24775898e-02 -2.49577343e-01  6.09844550e-02
 -3.09335172e-01  1.59978718e-01 -2.61880875e-01 -2.94375140e-02
  1.01137608e-01  1.63687877e-02 -1.55890107e-01  5.45244515e-01
 -3.23755264e-01 -2.37956583e-01 -2.52221197e-01  2.52246056e-02
 -5.19926250e-01  9.29039195e-02 -1.59072027e-01  8.78786519e-02
  2.12888300e-01  4.63810086e-01 -8.50785226e-02  3.28489542e-02
 -4.09064442e-01  8.12413469e-02 -4.57706392e-01 -1.82394475e-01
 -3.25937301e-01 -1.39067352e-01 -1.75929114e-01  1.17469147e-01
  2.10796483e-03  5.02661943e-01  2.32047334e-01  4.57966149e-01
 -2.86915861e-02  3.55693072e-01 -2.97135741e-01 -2.65016317e-01
 -2.87727974e-02  3.36934924e-01 -2.99857140e-01 -1.34413436e-01
  4.79250550e-02 -4.23789293e-01  4.12945524e-02  1.62133679e-01
  1.77837402e-01  2.21804053e-01 -4.04614985e-01 -1.66606426e-01
 -8.57127547e-01  9.00098383e-02 -2.96481520e-01 -1.30348474e-01
 -8.17409754e-02  2.53190577e-01 -3.26089412e-02  1.12150386e-01
  1.51856989e-03  4.27194387e-01 -2.33335048e-01  1.13821980e-02
 -7.06551690e-03  1.13897018e-01  2.34445985e-02  6.84075058e-05
 -1.51487157e-01  1.81355581e-01 -1.11728609e-01  5.16774133e-04
 -1.20903831e-03  1.89736843e-01 -2.36473039e-01  6.10814989e-02
  4.52656001e-02  3.14372361e-01 -5.08361384e-02  4.98991553e-03
 -9.03374255e-02  2.17197403e-01 -1.22452937e-01  6.11280240e-02
 -4.25978303e-01 -2.97768146e-01 -2.61652470e-01  1.24161271e-02
  3.91102135e-02  1.52499229e-01 -2.21227080e-01  2.75865626e-02
  4.36851420e-02  2.48558328e-01  1.15423761e-02  4.89675045e-01
 -1.56932354e-01  2.14897335e-01 -1.57205202e-03 -5.27021699e-02
  6.28904626e-02  3.24422151e-01 -3.89436901e-01 -4.45891917e-03
 -1.44775823e-01  1.74116224e-01  1.64318889e-01  1.05941817e-01
 -5.76544881e-01  3.91964257e-01  1.17898285e-01  4.29724157e-01
 -1.02643799e-02  3.34930658e-01  2.63334930e-01  2.06817195e-01
 -1.12270057e-01  4.52405870e-01 -5.72455581e-03  1.00485310e-01
 -3.81451249e-02  1.53637994e-02 -1.91496432e-01  2.64708877e-01
 -3.41633614e-03  1.86973050e-01  3.29475701e-02  2.50216126e-01
 -6.48499131e-02  3.02598238e-01 -1.53123066e-01  2.93923199e-01]

```