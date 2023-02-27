# Set working directory to the location of the script.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

dataset_path = "../data/archiveII/"
formatted_path = "../data/archiveII-arrays/"
rna_name = "5s_Acanthamoeba-castellanii-1"

# Load project code
import sys
sys.path.insert(1, '../src/utils/')
import datahandler

# Read a single CT file
title, bases, pairings = datahandler.read_ct(dataset_path + rna_name + ".ct")

print("Validating input file formatting.\n-----------------\n")

print("Result from a CT file:")
print(f"Title: {title}")
print(f"Bases: {''.join(bases)}")
print(pairings)
print(f"Secondary structure: {datahandler.get_dot_bracket_from_ct(pairings)}")
print()

# Tranform the sequences and pairings into a training-ready representation.
base_onehot = datahandler.sequence_to_one_hot(bases)
pairings_onehot = datahandler.pairings_to_one_hot(pairings)
print("Matrix representation:")
print(f"X: {base_onehot[0:4]} ...")
print(f"Y: {pairings_onehot[0:4]} ...")
print()

# Compare with formatted data
family = rna_name.split("_")[0]
with open(formatted_path + family + "_names.txt", 'r') as file:
    data = file.read()
    names = data.split('\n')
index = names.index(rna_name + ".ct")

import numpy as np
x = np.load(formatted_path + family + "_x.npy")[index]
y = np.load(formatted_path + family + "_y.npy")[index]

x = datahandler.remove_sequence_padding(x)
y = datahandler.remove_pairing_padding(y)

print("From the formatted dataset:")
print(f"Bases: {''.join(datahandler.one_hot_to_sequence(x))}")
print(f"Secondary structure: {''.join(datahandler.one_hot_to_pairing(y))}")
