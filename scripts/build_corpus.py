import glob
from pathlib import Path
import random

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
    for i in range(0, len(kmers), window_size):
        window = kmers[i:i + window_size]
        if len(window) < window_size and len(windows) > 0:
            # If the window size is less than the required size and there are previous windows
            missing_count = window_size - len(window)
            previous_window_kmers = windows[-1][-missing_count:]
            window.extend(previous_window_kmers[:missing_count])
        windows.append(window)
    return windows


def write_corpus(paths, output_path, sample_size, kmer_size, window_size, clear=True):
    if clear:
        with open(output_path, "w") as f:
            f.write('')  # Clearing the file if clear is True

    for path in paths:
        data = read_data(path, sample_size)
        kmers = [read2kmer(dna, kmer_size) for dna in data]
        windows = [kmer2window(kmer, window_size) for kmer in kmers]
        with open(output_path, "a") as f:
            for window_list in windows:
                for window in window_list:
                    f.write(" ".join(window) + "\n")

def __init__():
    main()

if __name__ == "__main__":
    paths = [str(x) for x in Path("..").glob("data/fasta/*/*.fasta")]
    write_corpus(paths, "../data/corpus/corpus.txt", 10000, kmer_size=7, window_size=10, clear=True)
