# Set working directory to the location of the script.
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

dataset_path = "../data/archiveII/"
formatted_dataset_path = "../data/archiveII-arrays/"

import sys
sys.path.insert(1, '../src/utils/')
import datahandler

def get_family_stats(filenames: str) -> tuple:
    lengths = []
    for filename in filenames:
        _, bases, _ = datahandler.read_ct(filename)
        if len(bases) <= 2048:
            lengths.append(len(bases))
    if not lengths:
        return None, None, None
    return np.mean(lengths), np.min(lengths), np.max(lengths)

def get_ct_stats(input: str) -> None:
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
        filename = file.name.split("/")[-1]
        family = filename.split("_")[0]
        if not family in families:
            families[family] = []
        families[family].append(name + extension)
    
    # Convert the file content to arrays and write them to files.
    for family in families:
        mean, min, max = get_family_stats(families[family])
        if mean:
            print(f"{family}: {mean:3.3f} [{min:3.3f}, {max:3.3f}]")
        else:
            print(f"{family} is empty.")

def get_array_stats(input: str) -> tuple:
    p = pathlib.Path(input)
    families = {}
    for file in p.iterdir():
        family = file.name.split("_")[0]
        families[family] = {"x": [], "y": []}
    for family in families:
        x = np.load(input + family + "_x.npy")
        y = np.load(input + family + "_y.npy")
        x_short = [datahandler.remove_sequence_padding(i) for i in x]
        y_short = [datahandler.remove_pairing_padding(i) for i in y]
        if x_short:
            families[family]['x'] = [len(i) for i in x_short]
            families[family]['y'] = [len(i) for i in y_short]
    return families

get_ct_stats(dataset_path)
families = (get_array_stats(formatted_dataset_path))

fig, ax = plt.subplots(3, 3, sharey=True)
i = 0
for family in families:
    lengths = families[family]['x']
    if lengths:
        subplot = ax[i % 3, int(i / 3)]
        subplot.hist(lengths, bins=min(len(lengths), 100))
        average = int(sum(lengths) / len(lengths))
        subplot.set_title(f"{family}: n={len(lengths)}, m={average}")
        subplot.set_ylim(0, max(lengths))
        i += 1
plt.tight_layout()
plt.show()
