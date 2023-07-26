![dna spacy logo](images/dna-spacy-logo.png)

# About

DNA spaCy is a machine learning [spaCy](www.spacy.io) pipeline for processing DNA sequences found in FASTA files. By treating the classification of DNA as an NLP problem, we can leverage NLP libraries, such as spaCy to classify DNA sequences. This methodology has already been explored by other [scholars](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680). To our knowledge, however, these methods have not been applied in a spaCy pipeline. We selected spaCy because of its wide use in industry and academia, speed, and its Python library is accessible and expandable.

The initial layer of the DNA spaCy layer is a custom tokenizer designed to work with DNA sequences represented as long sequences of unseparated letters that correspond to nucleotides. Traditionally, scientists divide these sequences into kmers, or substrings of length k. This is important for downstream tasks like classification, genome assembly, metagenomics, and many other applications. Here, the kmers correspond to "words" in the DNA sequence. The KMER tokenizer allows the user to pass a custom kmer size to tokenize the text. In addition to this, the KMER Tokenizer also converts the kmer sequences into unique integers. The KMERs are then subdivided into windows of a pre-specified length, which help alleviate the problem of DNA sequences of different lengths. These stages are important because they are the way in which our encoding layer, a transformer, was trained.

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
ACGGTCA CGGTCAT GGTCATC GTCATCA TCATCAT CATCATA ATCATAA TCATAAA CATAAAG ATAAAGG TAAAGGC AAAGGCA AAGGCAG AGGCAGT GGCAGTG GCAGTGA CAGTGAT AGTGATC GTGATCG TGATCGC GATCGCA ATCGCAC TCGCACC CGCACCC GCACCCT CACCCTC ACCCTCT CCCTCTG CCTCTGA CTCTGAC TCTGACA CTGACAG TGACAGA GACAGAC ACAGACG CAGACGT AGACGTC GACGTCA ACGTCAT CGTCATG GTCATGT TCATGTT CATGTTA ATGTTAA TGTTAAT GTTAATA 
Original DNA:
ACGGTCATCATAAAGGCAGTGATCGCACCCTCTGACAGACGTCATGTTAATA
DNA Vector:
[-1.28548875e-01 -2.38781303e-01  1.63492456e-01 -7.11313605e-01
  1.49529487e-01 -5.64375937e-01  1.76466480e-01 -2.07206495e-02
  5.60589917e-02 -1.11490563e-01  3.58448327e-01 -1.31397784e-01
  3.70010227e-01  4.97899354e-02 -3.84600982e-02 -8.21191818e-03
 -6.60514385e-02  7.06401020e-02 -3.06366980e-01  4.12903011e-01
  1.24116540e-01 -2.38114417e-01 -6.02178276e-02  7.94380531e-02
 -1.58490226e-01 -1.71716437e-01 -2.36673042e-01  1.04193315e-01
  8.50820243e-02  4.73545939e-02 -1.51460975e-01 -1.63663432e-01
 -7.31292441e-02 -2.97441810e-01  2.00450748e-01 -3.39161217e-01
 -3.26410413e-01  6.33492172e-02  3.10111791e-02 -7.01379329e-02
 -6.06193729e-02 -2.16402724e-01 -3.09330523e-01  1.81201734e-02
 -5.52482344e-02 -3.87324132e-02  1.98074915e-02  9.81337130e-02
  1.50880098e-01  4.34536338e-02 -1.84525758e-01 -2.67434925e-01
  5.05896993e-02 -1.97499990e-02  4.34214734e-02  1.81600004e-01
  1.55864507e-01  3.16447020e-01 -4.55345735e-02 -2.38655016e-01
  2.36718846e-03  2.73174644e-01  6.00490570e-01 -1.79090425e-01
  1.89047456e-01  3.78835917e-01 -1.09778389e-01  2.02044453e-02
 -1.12680010e-02  1.64930463e-01 -3.82675603e-03  6.02278486e-02
  1.52188599e-01  5.58185354e-02  3.72934073e-01  1.70346498e-01
  4.18234281e-02 -3.77320535e-02 -3.15801233e-01 -1.05514497e-01
 -1.91873997e-01  7.36140609e-02  1.97740778e-01 -1.23338766e-01
 -1.69717461e-01  3.69291395e-01  1.74139842e-01 -2.25363389e-01
 -1.52452707e-01  1.16280995e-01 -3.60330999e-01  1.93215394e-03
  3.44762087e-01  8.09920132e-02  1.83336526e-01 -5.16711250e-02
 -2.83963203e-01  1.17643028e-01  1.49079442e-01 -7.80047998e-02
  1.38652325e-01 -3.22632194e-02 -2.56561697e-01  1.19278878e-01
 -2.49606252e-01  2.01053731e-03  8.29940736e-02  6.73249960e-02
  6.30364418e-02 -4.17067707e-02 -8.50776210e-03 -1.51300758e-01
  8.01888630e-02  3.41205955e-01 -1.47559494e-01 -2.30541170e-01
 -4.72484007e-02  1.90698981e-01 -2.08991006e-01  1.30771905e-01
  2.37091482e-01  1.69885129e-01 -2.66904771e-01  5.98591492e-02
 -2.35189587e-01  9.22639817e-02 -2.45615572e-01  5.09769358e-02
 -3.13369364e-01  1.56338125e-01 -2.44901642e-01 -2.19766274e-02
  9.52866301e-02  1.48493927e-02 -1.62078962e-01  5.73674560e-01
 -3.48484039e-01 -2.32522726e-01 -2.25138366e-01  3.18090059e-02
 -4.90313679e-01  1.05107501e-01 -1.71310782e-01  1.03121281e-01
  1.93914473e-01  4.61949825e-01 -8.10789019e-02  3.61096114e-02
 -4.04098928e-01  8.36400613e-02 -4.50643599e-01 -1.74844801e-01
 -3.00849319e-01 -1.27129838e-01 -2.00468093e-01  1.22003406e-01
  5.11418097e-04  4.96835202e-01  2.22675532e-01  4.35725451e-01
 -3.48075107e-02  3.44941765e-01 -2.97615498e-01 -2.58587271e-01
 -3.15634646e-02  3.64515603e-01 -2.95336545e-01 -1.42283261e-01
  3.42165492e-02 -4.08767641e-01  5.86853921e-02  1.70663714e-01
  1.62758723e-01  1.87623844e-01 -3.93770874e-01 -1.51940405e-01
 -8.63100767e-01  8.75922889e-02 -2.91400313e-01 -1.26695603e-01
 -8.14299881e-02  2.63464183e-01 -3.08464915e-02  1.18302852e-01
  1.95044912e-02  4.41996425e-01 -2.38382623e-01  2.38629170e-02
 -2.05683745e-02  9.03545693e-02  3.34331989e-02  1.36588514e-02
 -1.70133993e-01  1.68786913e-01 -1.43423110e-01  6.37470745e-03
 -1.32120308e-02  1.91772684e-01 -2.16563910e-01  7.63974339e-02
  4.71089259e-02  3.00205171e-01 -2.06340477e-02  1.52869718e-02
 -8.43364447e-02  2.35612094e-01 -9.89452004e-02  6.80782571e-02
 -4.39307690e-01 -2.79908508e-01 -2.35124335e-01  1.70109165e-03
  1.59538146e-02  1.85054749e-01 -2.15571523e-01  5.27829155e-02
  1.52369039e-02  2.39193842e-01  2.92417035e-03  4.90627676e-01
 -1.34731725e-01  2.07111374e-01 -7.98024982e-03 -6.03660457e-02
  3.97653989e-02  3.49383891e-01 -3.70110452e-01 -4.17326689e-02
 -1.57052875e-01  1.64289579e-01  1.46018058e-01  1.12875260e-01
 -5.91504574e-01  3.75365496e-01  8.56786296e-02  3.92340630e-01
  2.86436267e-03  3.15959662e-01  2.73963153e-01  1.86017543e-01
 -1.21115826e-01  4.55716252e-01 -6.05055271e-03  7.68484101e-02
 -3.90151292e-02  2.28029024e-02 -1.77221626e-01  2.52177060e-01
 -1.54518746e-02  1.95084006e-01  5.96642941e-02  2.28341460e-01
 -3.69449966e-02  3.20325375e-01 -1.56883776e-01  3.13820034e-01]
```
