# Set working directory to the location of the script.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

dataset_path = "../data/archiveII/"
formatted_path = "../data/archiveII-arrays/"

# Load project code
import sys
sys.path.insert(1, '../src/utils/')
import datahandler

# Read a single CT file
title, bases, pairings = datahandler.read_ct(
    dataset_path + "5s_Acanthamoeba-castellanii-1.ct")

print("Result from the CT file:")
print(title)
print(''.join(bases))
print(datahandler.get_dot_bracket(pairings))
print()

# Read a single SEQ file
title, sequence = datahandler.read_seq(
    dataset_path + "5s_Acanthamoeba-castellanii-1.seq")

print("Result from the SEQ file:")
print(title)
print(sequence)
print()

# Tranform the sequences and pairings into a training-ready representation.
b = datahandler.sequence_to_one_hot(bases)
p = datahandler.pairings_to_one_hot(pairings)
print(b)
print(p)
print()

# Tranform the sequences and pairings into a training-ready representation.
print(datahandler.pad_one_hot_sequence(b, 512))
print(datahandler.pad_one_hot_pairing(p, 512))
print()

# Compare with formatted data
import numpy as np
x = np.load(formatted_path + "5s_x.npy")[575]
y = np.load(formatted_path + "5s_y.npy")[575]

print(datahandler.one_hot_to_sequence(x))

