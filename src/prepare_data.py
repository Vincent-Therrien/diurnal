# Set working directory to the location of the script.
import os
import shutil
import pathlib
import numpy as np
import utils.datahandler as datahandler

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

dataset_path = "../data/archiveII/"

#output_path = "../data/archiveII-shadows/"
#code = [0, 1, 1, 0]
#size = 512

#output_path = "../data/archiveII-structures/"
#code = [0, 1, -1, 0]
#size = 512

output_path = "../data/archiveII-classes/"
code = [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]]
size = 512

def format_family(filenames: str, max_size: int) -> tuple:
    X = []
    Y = []
    names = []
    for filename in filenames:
        x, y = datahandler.get_rna_x_y(filename, max_size, code)
        if not x or not y:
            continue
        X.append(x)
        Y.append(y)
        names.append(filename.split('/')[-1])
    return (np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32), names)

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
        name = name.replace("\\", "/") # Use / as the directory separator
        filename = name.split("/")[-1]
        family = filename.split("_")[0]
        if not family in families:
            families[family] = []
        families[family].append(name + extension)
    
    # Convert the file content to arrays and write them to files.
    for family in families:
        x, y, names = format_family(families[family], max_size)
        if len(names) < 1:
            print(f"Family {family} ({len(families[family])} samples) excluded.")
            continue
        base_name = output_path + family + "_"
        np.save(base_name + "x", x)
        np.save(base_name + "y", y)
        with open(base_name + "names.txt", "w") as outfile:
            outfile.write("\n".join(names))
        n_rejected = len(families[family]) - len(names)
        print(f"Wrote family {family} in files. " +
              f"{len(y)} files included. {n_rejected} files excluded.")

if os.path.isdir(output_path):
    shutil.rmtree(output_path)

os.makedirs(output_path)
format_archiveii(dataset_path, size)
