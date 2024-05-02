"""
    RNA secondary structure database utility module.

    This module contains functions to install (i.e. download and unwrap)
    RNA dataset and manipulate the data into matrix formats usable by
    processing algorithms. Note: the word `dataset` is used to refer to
    a given set of RNA secondary structures (e.g. archiveII or
    RNASTRalign). The collection of datasets is referred as the database.

    ::

       import diurnal.database as db
       db.download("./data/", "archiveII")
       db.format_basic("./data/archiveII", "./data/formatted", 512)

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import os
import pathlib
import numpy as np
from datetime import datetime
from typing import Callable
from random import shuffle
import json

from diurnal.utils import file_io, log
import diurnal.utils.rna_data as rna_data
import diurnal.structure
import diurnal.family

# Constant values.

_DB_IDs = {
    "RNA_STRAND.rst":     "1N4arnUYSgTkELcT7D43D-fUwFApd7Aee",
    "RNA_STRAND.tar.gz":  "1knqptKWhZLJRZgdX76KwWK3y94eoPIpI",
    "archiveII.rst":      "1yURblLBoBaJiW17lgnr6KflaIyNHrO84",
    "archiveII.tar.gz":   "1K4SJKsFngX1GTJtzh7NohCtoOZWGz2BQ",
    "RNASTRalign.rst":    "1PgwOk27mf8Uw0-zHffZfluuJeVinMnhw",
    "RNASTRalign.tar.gz": "1CHA9YxvaJ0Kb97B18A0RFH5-vzpIOoXu"
}
URL_PREFIX = "https://drive.google.com/uc?export=download&confirm=1&id="
DATA_FILE_ENDING = ".tar.gz"
INFO_FILE_ENDING = ".rst"
ALLOWED_DATASETS = {
    "archiveII": 3975,
    "RNASTRalign": 37149,
    "RNA_STRAND": 4666
}


# Installation functions.
def download(
        dst: str, datasets: list, cleanup: bool = True, verbosity: int = 1
        ) -> None:
    """Download and unpack RNA secondary structure databases.

    Download the datasets listed in the `datasets` argument, places them
    in the `dst` directory, and unpacks the downloaded files.

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
    if verbosity:
        log.info("Download and install an RNA database.")
    if dst[-1] != '/':
        dst += '/'
    if not os.path.exists(dst):
        os.makedirs(dst)
    # Data validation.
    if type(datasets) is str:
        datasets = [datasets]
    for dataset in datasets:
        if dataset not in ALLOWED_DATASETS:
            log.info(
                f"The dataset `{dataset}` is not allowed. "
                + f"Allowed databases are {ALLOWED_DATASETS}.")
            raise FileNotFoundError
    # Data obtention.
    for dataset in datasets:
        # Check if the data have already been downloaded.
        if file_io.is_downloaded(dst + dataset, ALLOWED_DATASETS[dataset]):
            if verbosity:
                log.trace(
                    f"The dataset `{dataset}` "
                    + f"is already downloaded at `{dst + dataset}`.")
            continue
        # Information file.
        file_name = dataset + INFO_FILE_ENDING
        url = URL_PREFIX + _DB_IDs[file_name]
        file_name = dst + file_name
        file_io.download(url, file_name, 0, dataset)
        # Data file.
        file_name = dataset + DATA_FILE_ENDING
        url = URL_PREFIX + _DB_IDs[file_name]
        file_name = dst + file_name
        file_io.download(url, file_name, verbosity, dataset)
        file_io.decompress(file_name, "r:gz", dst, verbosity, dataset)
        if cleanup:
            os.remove(file_name)
        if verbosity:
            log.info(f"Files installed in `{dst + dataset}`.")


def download_all(dst: str, cleanup: bool = True, verbosity: int = 1) -> None:
    """Download all available RNA secondary structure datasets
    (archiveII and RNASTRalign).

    Args:
        dst (str): Directory path in which the files are
            downloaded and unwrapped.
        cleanup (bool): If True, the raw, compressed file is deleted. If
            False, that file is not deleted.
        verbosity (int): Verbosity of the function. 1 (default) prints
            informative messages. 0 silences the function.
    """
    download(dst, ALLOWED_DATASETS, cleanup, verbosity)


def _get_structure_shape(
        max_size: int,
        path: str,
        primary_structure_map,
        secondary_structure_map) -> tuple:
    """Obtain the dimensions of formatted primary and secondary
    structures.

    Args:
        max_size (int): Maximum number of nucleotides.
        path (str): File path of a CT file to format.
        primary_structure_map
        secondary_structure_map

    Returns (tuple): Shapes formatted as (primary, secondary) or () if
        the structures are invalid.
    """
    length = rna_data.read_ct_file_length(str(path))
    if length > max_size:
        return (), ()
    _, bases, pairings = rna_data.read_ct_file(str(path))
    primary = primary_structure_map(bases, max_size)
    secondary = secondary_structure_map(pairings, max_size)
    return primary.shape, secondary.shape


def _is_already_encoded_basic(
        src: str, dst: str, size: int, primary_structure_map,
        secondary_structure_map) -> bool:
    """Check if a directory already contains already encoded data.

    The function makes the following verifications:
    - Ensure that the files for (1) the primary structure, (2) the
      secondary structure, (3) the names, (4) the families, and (5) the
      informative file are all present.
    - The data points are of expected dimensions.
    - The encoding for primary and secondary structures as well as
      families are formatted as expected.

    Args:
        src (str): Directory of the raw data files.
        dst (str): Directory of the formatted date files.
        size (int): Maximum size of a structure.
        primary_structure_map
        secondary_structure_map

    Returns (bool): True if data are formatted as expected, False
        otherwise.
    """
    # Expected files.
    if not os.path.isdir(dst):
        return False
    filenames = os.listdir(dst)
    expected_filenames = [
        'families.txt',
        'names.txt',
        'primary_structures.npy',
        'readme.rst',
        'secondary_structures.npy'
    ]
    if (filenames != expected_filenames):
        return False
    primary = np.load(dst + 'primary_structures.npy', mmap_mode='r')
    secondary = np.load(dst + 'secondary_structures.npy', mmap_mode='r')
    # Expected number of data points
    n_families = len(open(dst + 'families.txt').read().split('\n'))
    n_names = len(open(dst + 'names.txt').read().split('\n'))
    n_primary = primary.shape[0]
    n_secondary = secondary.shape[0]
    if n_families == n_names == n_primary == n_secondary:
        pass
    else:
        return False
    # Expected dimensions.
    if (primary.shape[1] != size or secondary.shape[1] != size):
        return False
    # Expected encoding.
    for path in pathlib.Path(src).rglob('*.ct'):
        _, bases, pairings = rna_data.read_ct_file(str(path))
        if len(bases) <= size:
            break
    test_primary = primary_structure_map(bases, size)
    test_secondary = secondary_structure_map(pairings, size)
    if (test_primary.tolist() != primary[0].tolist()
            or test_secondary.tolist() != secondary[0].tolist()):
        return False
    return True


def _mkdir(filename: str) -> None:
    """Safe-create a directory."""
    dir = "".join([i + "/" for i in filename.split("/")[:-1]])
    if not os.path.exists(dir):
        os.makedirs(dir)


def _format_metadata(filename: str, properties: dict) -> None:
    """Edit the file `metadata.json`."""
    name = filename.split("/")[-1]
    dir = "".join([i + "/" for i in filename.split("/")[:-1]])
    metadata_filename = dir + "metadata.json"
    metadata = {}
    if os.path.isfile(metadata_filename):
        with open(metadata_filename) as f:
            metadata = json.load(f)
    metadata["info"] = {}
    metadata["info"]["description"] = ("This file describes the format of "
        + "RNA structures formatted by the diurnal library.")
    metadata["info"]["Update time"] = str(datetime.now().isoformat())
    metadata[name] = properties
    with open(metadata_filename, 'w') as fp:
        json.dump(metadata, fp, indent=4)


def format_filenames(
        src: str,
        dst: str = None,
        size: int = 0,
        families: list[str] = [],
        randomize: bool = True,
        verbosity: int = 1
    ) -> list[str]:
    """Obtain all file names that satisfy the arguments.

    Args:
        src (str): Directory of the sequence files.
        dst (str): Output file name. Set to `None` for no output.
        size (int): Maximum length of a sequence. Provide `0` for no
            maximum length.
        families (list[str]): Set of RNA families to include. Provide
            `[]` to include all families.
        randomize (bool): If True, shuffle the filenames.
        verbosity (int): Verbosity level. `0` to disable the output.

    Returns (list[str]): List of file names.
    """
    _mkdir(dst)
    if verbosity:
        log.info(f"Extract the filenames from the directory `{src}`.")
    data = []
    total = 0
    paths = pathlib.Path(src).rglob('*.ct')
    for i in paths:
        total += 1
    paths = pathlib.Path(src).rglob('*.ct')
    for i, path in enumerate(paths):
        family = diurnal.family.get_name(str(path))
        if families and not family in families:
            pass
        elif rna_data.read_ct_file_length(path) > size:
            pass
        else:
            data.append(str(path))
        if verbosity:
            suffix = f" {path.name}"
            log.progress_bar(total, i, suffix)
    if verbosity:
        print()
        log.trace(f"Detected {total} files. Kept {len(data)} files.")
    if dst and os.path.exists(dst):
        with open(dst, "r") as file:
            lines = [line.rstrip() for line in file]
        if set(lines) == set(data):
            if verbosity:
                log.trace(f"The file `{dst}` already contains the names.")
            return lines
    if randomize:
        if verbosity:
            log.trace(f"Shuffling data.")
        shuffle(data)
    if dst:
        if verbosity:
            log.trace(f"Writing data in the file {dst}")
        with open(dst, "w") as file:
            file.write("\n".join(data))
        _format_metadata(dst,
            {
                "Input directory": src,
                "Generation time": str(datetime.utcnow()),
                "Number of file paths": len(data),
                "Randomized": randomize,
                "families": families
            }
        )
    return data


def _is_already_encoded(
        structure_type: str,
        names: list[str],
        dst: str,
        size: int,
        map: Callable
    ) -> bool:
    """Check if the `dst` file is already formatted."""
    # File existence.
    if not os.path.exists(dst):
        return False
    # Number of data points.
    N = len(names)
    array = np.load(dst, mmap_mode='r')
    if N != array.shape[0]:
        return False
    # Expected encoding.
    for name in names:
        _, bases, _ = rna_data.read_ct_file(name)
        break
    _, bases, pairings = rna_data.read_ct_file(name)
    if structure_type == "primary":
        input = bases
    elif structure_type == "secondary":
        input = pairings
    test_array = map(input, size)
    if (test_array.tolist() != array[0].tolist()):
        return False
    return True


def _format_structure(
        structure_type: str,
        names: list[str],
        dst: str,
        size: int,
        map: Callable,
        verbosity: int = 1
    ) -> None:
    """Convert structures into a Numpy file.

    Args:
        names (list[str]): List of sequence file names.
        dst (str): Output file name.
        size (int): Maximum length of a sequence.
        map (Callable): Function that transforms the sequence of bases
            into a formatted primary structure.
        verbosity (int): Verbosity level. `0` to disable the output.
    """
    # Preparation: Check the existence and validity of the file.
    _mkdir(dst)
    # Add file names if not present.
    dir = "".join([i + "/" for i in dst.split("/")[:-1]])
    files = os.listdir(dir)
    if f"names.txt" in files:
        with open(f"{dir}/names.txt", "r") as file:
            lines = [line.rstrip() for line in file]
    else:
        lines = []
    if set(lines) != set(names):
        with open(f"{dir}/names.txt", "w") as file:
            file.write("\n".join(names))
        _format_metadata(f"{dir}/names.txt",
            {
                "Generation time": str(datetime.utcnow()),
                "Number of file paths": len(names)
            }
        )
    if verbosity:
        log.info(f"Formatting {structure_type} structures into `{dst}`.")
    if _is_already_encoded(structure_type, names, dst, size, map):
        log.trace(f"The file `{dst}` already contains the formatted data.")
        return
    # Determine the shape of the data.
    _, bases, pairings = rna_data.read_ct_file(names[0])
    if structure_type == "primary":
        input = bases
    elif structure_type == "secondary":
        input = pairings
    else:
        log.error(f"Invalid structure type: {structure_type}")
        raise RuntimeError
    shape = (len(names), ) + map(input, size).shape
    # Open a memory mapped file and store the data.
    file = np.lib.format.open_memmap(
        dst, dtype='float32', mode='w+', shape=shape
    )
    for i, name in enumerate(names):
        _, bases, pairings = rna_data.read_ct_file(name)
        if structure_type == "primary":
            input = bases
        elif structure_type == "secondary":
            input = pairings
        file[i] = map(input, size)
        file.flush()
        if verbosity:
            suffix = f" {name.split('/')[-1]}"
            log.progress_bar(len(names), i, suffix)
    if verbosity:
        print()  # Change the line after the progress bar.
        log.trace(f"Encoded {len(names)} files.")
    _format_metadata(dst,
        {
            "Generation time": str(datetime.now().isoformat()) + " UTC",
            "Array shape": shape,
            "Data type": "float32",
            "Structure type": structure_type
        }
    )


def format_primary_structure(
        names: str,
        dst: str,
        size: int,
        map: Callable,
        verbosity: int = 1
    ) -> str:
    """Convert primary structures into a Numpy file.

    Args:
        names (list[str]): List of sequence file names.
        dst (str): Output file name.
        size (int): Maximum length of a sequence.
        map (Callable): Function that transforms the sequence of bases
            into a formatted primary structure.
        verbosity (int): Verbosity level. `0` to disable the output.
    """
    _format_structure("primary", names, dst, size, map, verbosity)


def format_secondary_structure(
        names: str,
        dst: str,
        size: int,
        map: Callable,
        verbosity: int = 1
    ) -> str:
    """Convert secondary structures into a Numpy file.

    Args:
        names (list[str]): List of sequence file names.
        dst (str): Output file name.
        size (int): Maximum length of a sequence.
        map (Callable): Function that transforms the sequence of bases
            into a formatted primary structure.
        verbosity (int): Verbosity level. `0` to disable the output.
    """
    _format_structure("secondary", names, dst, size, map, verbosity)


def format_basic(
        src: str,
        dst: str,
        max_size: int,
        primary_structure_map: any = diurnal.structure.Primary.to_onehot,
        secondary_structure_map: any = diurnal.structure.Secondary.to_onehot,
        verbosity: int = 1) -> None:
    """Transform the original datasets into the representation provided
    by the arguments.

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
        verbosity (int): Verbosity level of the function. 1 (default)
            prints informative messages. 0 silences the function.
    """
    if verbosity:
        log.info("Format RNA data into Numpy files.")
    if dst[-1] != '/':
        dst += '/'
    # If the data is already encoded in the directory, exit.
    if _is_already_encoded_basic(
            src, dst, max_size, primary_structure_map,
            secondary_structure_map):
        log.trace(f"The directory {dst} already contains the formatted data.")
        return
    # Create the directory if it des not exist.
    if not os.path.exists(dst):
        os.makedirs(dst)
    # Obtain the list of files to read.
    paths = []
    x_shape, y_shape = (), ()
    n_samples = 0
    for path in pathlib.Path(src).rglob('*.ct'):
        paths.append(path)
        if not x_shape:
            x_shape, y_shape = _get_structure_shape(
                max_size, path, primary_structure_map, secondary_structure_map)
        if rna_data.read_ct_file_length(path) <= max_size:
            n_samples += 1
    x_shape = (n_samples, ) + x_shape
    y_shape = (n_samples, ) + y_shape
    # Encode the content of each file.
    names = []           # RNA molecule names included in the dataset.
    rejected_names = []  # RNA molecule names excluded from the dataset.
    families = []        # Family
    X_file = np.lib.format.open_memmap(  # Primary structure
        dst + "primary_structures.npy",
        dtype='float32', mode='w+', shape=x_shape)
    Y_file = np.lib.format.open_memmap(  # Secondary structure
        dst + "secondary_structures.npy",
        dtype='float32', mode='w+', shape=y_shape)
    offset = 0
    for i, path in enumerate(paths):
        if rna_data.read_ct_file_length(str(path)) > max_size:
            rejected_names.append(str(path))
            continue
        _, bases, pairings = rna_data.read_ct_file(str(path))
        family = diurnal.family.get_name(str(path))
        names.append(str(path))
        families.append(family)
        X_file[offset] = primary_structure_map(bases, max_size)
        X_file.flush()
        Y_file[offset] = secondary_structure_map(pairings, max_size)
        Y_file.flush()
        offset += 1
        if verbosity:
            suffix = f" {path.name}"
            log.progress_bar(n_samples, offset, suffix)
    if verbosity:
        print()  # Change the line after the progress bar.
        i = len(names)
        r = len(rejected_names)
        log.trace(f"Encoded {i} files. Rejected {r} files.")
    # Write the encoded file content into Numpy files.
    filename = dst + "families.txt"
    if verbosity:
        log.trace(f"Writing families at `{filename}`.")
    with open(filename, "w") as outfile:
        outfile.write("\n".join(families))
    filename = dst + "names.txt"
    if verbosity:
        log.trace(f"Writing names at `{filename}`.")
    with open(filename, "w") as outfile:
        outfile.write("\n".join(names))
    # Write an informative file to sum up the content of the formatted folder.
    info = dst + "readme.rst"
    if verbosity:
        log.trace(f"Writing an informative file at `{info}`.")
    with open(info, "w") as outfile:
        outfile.write(summarize(
            dst, primary_structure_map,
            secondary_structure_map))


def summarize(
        path: str,
        primary_structure_map,
        secondary_structure_map) -> str:
    """Summarize the content of the formatted file directory.

    Args:
        path (str): File path of the formatted data.
        primary_structure_map: A dictionary or function that maps
            an RNA primary structure symbol to a vector (e.g. map A to
            [1, 0, 0, 0]). If None, the file `x.np` is not written.
        secondary_structure_map: A dictionary or function that maps
            an RNA secondary structure symbol to a vector (e.g. map '.'
            to [0, 1, 0]). If None, the file `y.np` is not written.

    Returns (str): Informative file containing:
        - Title
        - Generation date and time
        - Number of structures
        - Structure size (number of nucleotides)
        - Primary structure encoding example
        - Secondary structure encoding example
    """
    content = "[> DIURNAL] RNA Database File Formatting\n"
    content += "========================================\n\n"
    content += f"Generation timestamp: {datetime.utcnow()} UTC\n\n"
    X = np.load(path + "primary_structures.npy", mmap_mode='r')
    content += f"Number of structures: {X.shape[0]}\n\n"
    content += "RNA molecule **names** are listed in `names.txt`.\n\n"
    content += "RNA molecule **families** are listed in `families.txt`.\n\n"
    if X.any():
        content += "Primary Structure Encoding\n"
        content += "--------------------------\n\n"
        content += f"File: `{path + 'primary_structures.npy'}`\n\n"
        content += f"Shape: {X.shape}\n\n"
        example = "ACGU-"
        content += f"Encoding of the structure `{example}`::\n\n"
        code = primary_structure_map(example)
        for i in range(len(example)):
            content += f"   {example[i]} -> {code[i]}\n"
    Y = np.load(path + "secondary_structures.npy", mmap_mode='r')
    if Y.any():
        content += "Secondary Structure Encoding\n"
        content += "----------------------------\n\n"
        content += f"File: `{path + 'secondary_structures.npy'}`\n\n"
        content += f"Shape: {Y.shape}\n\n"
        example = [2, -1, 0]  # Corresponds to `(.)` in bracket notation.
        content += f"Encoding of the structure `{example}`::\n\n"
        code = secondary_structure_map(example)
        for i in range(len(example)):
            content += f"   {example[i]} -> {code[i]}\n"
    return content
