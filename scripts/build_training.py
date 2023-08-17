import glob
import argparse
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import srsly
import spacy
from spacy.tokens import DocBin

def read_data(path, sample_size, seed_value=1):
    rng = random.Random(seed_value)
    with open(path, "r") as f:
        data = f.read()
    reads = data.split(">")
    reads = ["".join(read.split("\n")[1:]) for read in reads]
    rng.shuffle(reads)
    reads = reads[:sample_size+1]
    return reads

def read2kmer(dna, kmer_size):
    return [dna[i:i+kmer_size] for i in range(len(dna) - kmer_size + 1)]

def kmer2window(kmers, window_size):
    windows = []
    kmers_length = len(kmers)
    for i in range(0, kmers_length - window_size + 1, window_size):
        windows.append(" ".join(kmers[i:i + window_size]))
    if kmers_length % window_size != 0:
        windows.append(" ".join(kmers[-window_size:]))
    if len(windows) == 0:
        print("Window Size is too small.")
    return windows


def write_corpus(paths, sample_size, kmer_size, window_size, clear=True):
    training_data = []
    unique_labels = set()

    # First, discover all unique labels
    for path in paths:
        label_dir = Path(path).parts[-2].upper()
        unique_labels.add(label_dir)

    # Now, process the data and set labels
    for path in paths:
        label_dir = Path(path).parts[-2].upper()
        data = read_data(path, sample_size)
        kmers = [read2kmer(dna, kmer_size) for dna in data]
        windows = [kmer2window(kmer, window_size) for kmer in kmers]

        # Create a label dictionary with all unique labels set to 0.0
        label = {lbl: 0.0 for lbl in unique_labels}
        # Set the current label to 1.0
        label[label_dir] = 1.0

        for window_list in windows:
            for window in window_list:
                training_data.append({"text": window, "cats": label.copy()}) # Use copy to avoid reference issues

    train, valid = train_test_split(training_data, random_state=1)
    srsly.write_jsonl("assets/train.jsonl", train) # Write as JSONL
    srsly.write_jsonl("assets/dev.jsonl", valid)   # Write as JSONL


def text2spacy(input_path, output_path):
    nlp = spacy.blank("en")
    db = DocBin()

    data = srsly.read_jsonl(input_path) # Read JSONL file
    for line in data:
        doc = nlp.make_doc(line["text"])
        doc.cats = line["cats"] # Use the labels stored in the "cats" attribute
        db.add(doc)
    db.to_disk(output_path)
    print(f"Documents saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process DNA sequences')
    parser.add_argument('reads', type=int, help='Amount of DNA sequence reads')
    parser.add_argument('kmer_size', type=int, help='Kmer size')
    parser.add_argument('window_size', type=int, help='Window size')
    parser.add_argument('--clear', action='store_true', default=True, help='Clear the file (default is True)')
    
    
    args = parser.parse_args()

    paths = [str(x) for x in Path("").glob("assets/fasta/*/*.fasta")]
    print(paths)
    write_corpus(paths, args.reads, args.kmer_size, args.window_size, args.clear)

    text2spacy("assets/train.jsonl", "assets/train.spacy")
    text2spacy("assets/dev.jsonl", "assets/dev.spacy")

if __name__ == "__main__":
    main()
