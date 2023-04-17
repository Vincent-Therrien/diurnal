"""
    RNA secondary structure database utility module.

    This module contains functions to install (i.e. download and unwrap)
    RNA dataset and manipulate the data into matrix formats usable by
    processing algorithms. Note: the word `dataset` is used to refer to
    a given set of RNA secondary structures (e.g. archiveII or
    RNASTRalign). The collection of datasets is refered as the database.

    Example:
        blocks:: python

            import diurnal.database
            from diurnal.encoding import PrimaryStructure as s1
            from diurnal.encoding import SecondaryStructure as s2
            
            diurnal.database.download_all("./data/")
            diurnal.database.format(
                "./data/", # Raw data input directory.
                "./data/formatted", # Formatted data output directory.
                s1.iupac_onehot, # RNA encoding scheme.
                s2.bracket_onehot, # RNA encoding scheme.
            )
            diurnal.database.visualize("./data/formatted")

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import os
import pathlib
import numpy as np
import inspect
from datetime import datetime

from .utils import file_io

# Constant values.
URL_PREFIX ="https://github.com/Vincent-Therrien/rna-2s-database/raw/main/data/"
FILE_ENDING = ".tar.gz"
ALLOWED_DATASETS = [
    "archiveII",
    "RNASTRalign"
]
FAMILIES = [
    ["5s"],  # 5s ribosomal RNA (rRNA)
    ["16s"], # 16s ribosomal RNA (rRNA)
    ["23s"], # 23s ribosomal RNA (rRNA)
    ["grp1", "group_I_introns"],
    ["grp2", "group_II_introns"],
    ["RNaseP"],
    ["SRP"],
    ["telomerase"],
    ["tmRNA"],
    ["tRNA"]
]

# Installation functions.
def download(dst: str, datasets: list, cleanup: bool=True, verbosity: int=1
             ) -> None:
    """
    Download and unpack RNA secondary structure databases.

    This function downloads the datasets listed in the `datasets`
    argument, places them in the `dst` directory, and unpacks
    the downloaded files.

    Args:
        dst (str): Directory path in which the files are downloaded and
            unwrapped.
        datasets (list(str)): The list of databases to download. The
            allowed databases are `archiveII` and `RNASTRalign`.
        cleanup (bool): If True, the raw, compressed file is deleted. If
            False, that file is not deleted.
        verbosity (int): Verbosity of the function. 1 (default) prints
            informative messages. 0 silences the function.
    """
    if verbosity: file_io.log("Download and install an RNA database.")
    if dst[-1] != '/':
        dst += '/'
    # Data validation.
    if type(datasets) is str:
        datasets = [datasets]
    for dataset in datasets:
        if dataset not in ALLOWED_DATASETS:
            file_io.log(f"The dataset `{dataset}` is not allowed. "
                + f"Allowed databases are {ALLOWED_DATASETS}.")
            raise FileNotFoundError
    # Data obtention.
    for dataset in datasets:
        url = URL_PREFIX + "/" + dataset + FILE_ENDING
        file_name = dst + dataset + FILE_ENDING
        file_io.download(url, file_name, verbosity, dataset)
        file_io.decompress(file_name, "r:gz", dst, verbosity, dataset)
        if cleanup:
            os.remove(file_name)
        if verbosity:
            file_io.log(f"Files installed in the directory `{dst + dataset}`.")

def download_all(dst: str, cleanup: bool=True, verbosity: int=1) -> None:
    """
    Download all available RNA secondary structure datasets (archiveII
    and RNASTRalign).

    Args:
        dst (str): Directory path in which the files are
            downloaded and unwrapped.
        cleanup (bool): If True, the raw, compressed file is deleted. If
            False, that file is not deleted.
        verbosity (int): Verbosity of the function. 1 (default) prints
            informative messages. 0 silences the function.
    """
    download(dst, ALLOWED_DATASETS, cleanup, verbosity)

def get_rna_family(path: str) -> str:
    """
    Determine the RNA family from the file path. For archiveII, the
    family is indicated in the file name. For RNASTRalign, it is
    indicated is the directory name.
    
    Args:
        path (str): Complete name of the file.
    """
    candidates = []
    for family in FAMILIES:
        for alias in family:
            if alias.upper() in path.upper():
                candidates.append(family[0])
                break
    if not candidates:
        file_io.log(f"Unknown family for `{path}`.", -1)
        raise RuntimeError
    if len(candidates) == 1:
        return candidates[0]
    # If several families are comprised in the file name, return the last one.
    else:
        c = {}
        for candidate in candidates:
            c[path.upper().find(candidate.upper())] = candidate
        return c[max(c.keys())]

def _encode_primary_structure(bases: list, target_size: int, map) -> tuple:
    """
    Encode a primary structure into a matrix.

    Args:
        bases list(str): List of nucleotides (i.e. 'A', 'C', 'G', 'U').
        target_size (int): Size of the encoded structure. Padding is
            determined by the `map` argument.
        map: A function that maps a base pair to a value.
    """
    if inspect.isfunction(map):
        return map(bases, target_size)
    else:
        message = (f"Type `{type(map)}` is not allowed for primary "
            + " structure encoding. Use a mapping function instead, "
            + "e.g. `diurnal.encoding.PrimaryStructure.iupac_onehot(bases)`.")
        file_io.log(message, -1)
        raise RuntimeError

def _encode_secondary_structure(pairings: list, target_size: int, map) -> tuple:
    """
    Encode a secondary structure into a matrix.

    Args:
        pairings list(int): Paired nucleotide index, -1 is unpaired.
        target_size (int): Size of the encoded structure. Padding is
            determined by the `map` argument.
        map: A function that maps a base pair to a value.
    """
    if inspect.isfunction(map):
        return map(pairings, target_size)
    else:
        message = (f"Type `{type(map)}` is not allowed for secondary "
            + " structure encoding. Use a mapping function instead, "
            + "e.g. `diurnal.encoding.PrimaryStructure.iupac_onehot(bases)`.")
        file_io.log(message, -1)
        raise RuntimeError

def _encode_family(family: str, map) -> tuple:
    """
    Encode an RNA family into a matrix.

    Args:
        family (str): Family name.
        map: A function that maps a family to a value.
    """
    if inspect.isfunction(map):
        return map(family)
    else:
        message = (f"Type `{type(map)}` is not allowed for family encoding. "
            + "Use a mapping function instead, "
            + "e.g. `diurnal.encoding.Family.onehot(bases)`.")
        file_io.log(message, -1)
        raise RuntimeError

def format(src: str,
          dst: str,
          max_size: int,
          primary_structure_map,
          secondary_structure_map,
          family_map=None,
          verbosity: int=1) -> None:
    """
    Transform the original datasets into the representation provided by
    the arguments.

    This function reads the RNA dataset files comprised in the directory
    `dataset_path`, applies the encoding schemes defined by the
    arguments, and writes the result in the `formatted_path` directory.
    All encoded elements are zero-padded to obtain elements of
    dimensions [1 X max_size].

    The function writes four files:
    - `info.rst` describes the data.
    - `primary_structure.np` contains the encoded primary structures of
        the molecules.
    - `secondary_structure.np` contains the encoded secondary structures
        of the molecules.
    - `families.np` contains the encoded family of the molecules.
    - `names.txt` contains the newline-delimited names of the molecules.

    Args:
        src (str): The directory in which RNA datasets are located. The
            function searches for RNA files recursively.
        dst (str): The directory in which the encoded RNA structures
            are written. If the directory does not exist, it is created.
        max_size (int): Maximal number of nucleotides in an RNA
            structure. If an RNA structure has more nucleotides than
            `max_size`, it is not included in the formatted dataset.
        primary_structure_map: A dictionary or function that maps
            an RNA primary structure symbol to a vector (e.g. map A to
            [1, 0, 0, 0]). If None, the file `x.np` is not written.
        secondary_structure_map: A dictionary or function that maps
            an RNA secondary structure symbol to a vector (e.g. map '.'
            to [0, 1, 0]). If None, the file `y.np` is not written.
        family_map: A dictionary or function that maps an RNA
            family name (e.g. '5s') to a vector (e.g. '[1, 0, 0]).
            If None, the file `family.np` is not written.
        verbosity (int): Verbosity level of the function. 1 (default)
            prints informative messages. 0 silences the function.
    """
    if verbosity: file_io.log("Encode RNA data into Numpy files.")
    # Create the directory if it des not exist.
    if dst[-1] != '/': dst += '/'
    if not os.path.exists(dst):
        os.makedirs(dst)
    # Obtain the list of files to read.
    paths = []
    for path in pathlib.Path(src).rglob('*.ct'):
        paths.append(path)
    # Encode the content of each file.
    names = [] # RNA molecule names included in the dataset.
    rejected_names = [] # RNA molecule names excluded from the dataset.
    X = [] # Primary structure
    Y = [] # Secondary structure
    F = [] # Family
    for i, path in enumerate(paths):
        # Read the file.
        _, bases, pairings = file_io.read_ct_file(str(path))
        family = get_rna_family(str(path))
        # Add the file to the dataset or not depending on its size.
        if len(bases) > max_size:
            rejected_names.append(str(path))
            continue
        else:
            names.append(str(path))
        # Encode the data.
        if primary_structure_map:
            X.append(_encode_primary_structure(bases, max_size,
                                               primary_structure_map))
        if secondary_structure_map:
            Y.append(_encode_secondary_structure(pairings, max_size,
                                                 secondary_structure_map))
        if family_map:
            F.append(_encode_family(family, family_map))
        if verbosity:
            prefix = f"    Encoding {len(paths)} files "
            suffix = f" {path.name}"
            file_io.progress_bar(len(paths), i, prefix, suffix)
    if verbosity:
        print() # Change the line after the progress bar.
        i = len(names)
        r = len(rejected_names)
        file_io.log(f"Encoded {i} files. Rejected {r} files.", 1)
    # Write the encoded file content into Numpy files.
    if X:
        if verbosity: file_io.log(f"Writing primary structures.", 1)
        np.save(dst + "primary_structures", np.asarray(X, dtype=np.float32))
    if Y:
        if verbosity: file_io.log("Writing secondary structures.", 1)
        np.save(dst + "secondary_structures", np.asarray(Y, dtype=np.float32))
    if F:
        if verbosity: file_io.log("Writing families.", 1)
        np.save(dst + "families", np.asarray(F, dtype=np.float32))
    if names:
        if verbosity: file_io.log("Writing names.", 1)
        with open(dst + "names.txt", "w") as outfile:
            outfile.write("\n".join(names))
    # Write an informative file to sum up the content of the formatted folder.
    with open(dst + "info.rst", "w") as outfile:
        outfile.write(summarize(dst, primary_structure_map,
                                secondary_structure_map, family_map))

def summarize(path: str,
              primary_structure_map,
              secondary_structure_map,
              family_map) -> str:
    """
    Summarize the content of the formatted file directory.

    Args:
        path (str): File path of the formatted data.
    
    Returns (str): Informative file containing:
        - Title
        - Generation date and time
        - Number of structures
        - Structure size (number of nucleotides)
        - Primary structure encoding example
        - Secondary structure encoding example
        - Family encoding example
    """
    content =  "[> DIURNAL] RNA Database File Formatting\n"
    content += "========================================\n\n"
    content += f"Generation timestamp: {datetime.utcnow()} UTC\n\n"
    X = np.load(path + "primary_structures.npy")
    content += f"Number of structures: {X.shape[0]}\n\n\n"
    if X.any():
        content += "Primary Structure Encoding\n"
        content += "--------------------------\n\n"
        content += f"File: `{path + 'primary_structures.npy'}`\n\n"
        content += f"Shape: {X.shape}\n\n"
        content += "Encoding:\n"
        example = "ACGU-"
        code = _encode_primary_structure(example, 0, primary_structure_map)
        for i in range(len(example)):
            content += f"    {example[i]} -> {code[i]}\n"
        content += "\nExample:\n"
        content += str(X[0])
        content += "\n\n\n"
    Y = np.load(path + "secondary_structures.npy")
    if Y.any():
        content += "Secondary Structure Encoding\n"
        content += "----------------------------\n\n"
        content += f"File: `{path + 'secondary_structures.npy'}`\n\n"
        content += f"Shape: {Y.shape}\n\n"
        content += "Encoding:\n"
        example = [2, -1, 0] # Corresponds to `(.)` in bracket notation.
        code =_encode_secondary_structure(example,4,secondary_structure_map)
        for i in range(len(example)):
            content += f"    {example[i]} -> {code[i]}\n"
        content += "\nExample:\n"
        content += str(Y[0])
        content += "\n\n\n"
    F = np.load(path + "families.npy")
    if F.any():
        content += "Family Encoding\n"
        content += "---------------\n\n"
        content += f"File: `{path + 'families.npy'}`\n\n"
        content += f"Shape: {F.shape}\n\n"
        content += "Encoding:\n"
        for f in FAMILIES:
            content+=f"    {f[0]} -> {_encode_family(f[0], family_map)}"
            content += "\n"
        content += "\nExample: \n"
        content += str(F[0])
        content += "\n"
    return content

def visualize(path: str):
    """
    """
    pass
