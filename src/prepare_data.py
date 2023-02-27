# Set working directory to the location of the script.
import os
import shutil
import pathlib
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

dataset_path = "../data/archiveII/"
output_path = "../data/archiveII-arrays/"

# Load project code
import sys
sys.path.insert(1, './utils/')
import datahandler

def format_family(filenames: str, max_size: int) -> tuple:
    X = []
    Y = []
    for filename in filenames:
        x, y = datahandler.get_rna_x_y(filename, max_size)
        if not x or not y:
            continue
        X.append(x)
        Y.append(y)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

def format_archiveii(input: str, max_size: int) -> None:
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
        x, y = format_family(families[family], max_size)
        base_name = output_path + family + "_"
        np.save(base_name + "x", x)
        np.save(base_name + "y", y)
        names = [f.split("/")[-1] for f in families[family]]
        with open(base_name + "names.txt", "w") as outfile:
            outfile.write("\n".join(names))
        print(f"Wrote family {family} in files.")

if os.path.isdir(output_path):
    shutil.rmtree(output_path)

os.makedirs(output_path)
format_archiveii(dataset_path, 512)
