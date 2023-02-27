# Set working directory to the location of the script.
import os
import pathlib
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

dataset_path = "../data/archiveII/"

# Load project code
import sys
sys.path.insert(1, '../src/utils/')
import datahandler

def get_family_stats(filenames: str) -> tuple:
    lengths = []
    for filename in filenames:
        _, bases, _ = datahandler.read_ct(filename)
        if len(bases) <= 512:
            lengths.append(len(bases))
    return np.mean(lengths)

def get_dataset_stats(input: str) -> None:
    # Fetch all file names
    p = pathlib.Path(input)
    families = {}
    for file in p.iterdir():
        # Select CT files
        if file.is_dir():
            continue
        name, extension = os.path.splitext(file)
        if extension != ".ct":
            continue
        # Determine the RNA family
        filename = name.split("/")[-1]
        family = filename.split("_")[0]
        if not family in families:
            families[family] = []
        families[family].append(name + extension)
    
    # Convert the file content to arrays and write them to files.
    for family in families:
        mean = get_family_stats(families[family])
        print(f"{family}: {mean}")

get_dataset_stats(dataset_path)
