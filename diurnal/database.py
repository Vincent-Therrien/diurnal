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
       db.format("./data/archiveII", "./data/formatted", 512)

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2023
    - License: MIT
"""

import os
import pathlib
import numpy as np
from datetime import datetime

from diurnal.utils import file_io, log
import diurnal.utils.rna_data as rna_data
import diurnal.structure
import diurnal.family

# Constant values.
URL_PREFIX = \
    "https://github.com/Vincent-Therrien/rna-2s-database/raw/main/data/"
DATA_FILE_ENDING = ".tar.gz"
INFO_FILE_ENDING = ".rst"
ALLOWED_DATASETS = {
    "archiveII": 3975,
    "RNASTRalign": 37149,
    "RNA_STRAND": 4666
}


# Installation functions.
def available_datasets() -> None:
    """Print available RNA datasets."""
    log.info(f"Available datasets: {ALLOWED_DATASETS}")


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
        url = URL_PREFIX + "/" + dataset + INFO_FILE_ENDING
        file_name = dst + dataset + INFO_FILE_ENDING
        file_io.download(url, file_name, 0, dataset)
        # Data file.
        url = URL_PREFIX + "/" + dataset + DATA_FILE_ENDING
        file_name = dst + dataset + DATA_FILE_ENDING
        file_io.download(url, file_name, verbosity, dataset)
        file_io.decompress(file_name, "r:gz", dst, verbosity, dataset)
        if cleanup:
            os.remove(file_name)
        if verbosity:
            log.info(f"Files installed in `{dst + dataset}`.", 1)


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


def _is_already_encoded(
        src: str, dst: str, size: int, primary_structure_map,
        secondary_structure_map, family_map) -> bool:
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
        family_map

    Returns (bool): True if data are formatted as expected, False
        otherwise.
    """
    # Expected files.
    filenames = os.listdir(dst)
    expected_filenames = [
        'families.npy',
        'info.rst',
        'names.txt',
        'primary_structures.npy',
        'secondary_structures.npy'
    ]
    if (filenames != expected_filenames):
        return False
    # Expected dimensions.
    primary = np.load(dst + 'primary_structures.npy')
    secondary = np.load(dst + 'secondary_structures.npy')
    family = np.load(dst + 'families.npy')
    if (primary.shape[1] != size or secondary.shape[1] != size):
        return False
    # Expected encoding.
    for path in pathlib.Path(src).rglob('*.ct'):
        _, bases, pairings = rna_data.read_ct_file(str(path))
        if len(bases) <= size:
            test_family_name = diurnal.family.get_name(str(path))
            break
    test_primary = primary_structure_map(bases, size)
    test_secondary = secondary_structure_map(pairings, size)
    test_family = family_map(test_family_name)
    if (list(test_family) != list(family[0])
            or test_primary.tolist() != primary[0].tolist()
            or test_secondary.tolist() != secondary[0].tolist()):
        return False
    return True


def format(
        src: str,
        dst: str,
        max_size: int,
        primary_structure_map: any = diurnal.structure.Primary.to_vector,
        secondary_structure_map: any = diurnal.structure.Secondary.to_vector,
        family_map: any = diurnal.family.to_vector,
        verbosity: int = 1) -> None:
    """ Transform the original datasets into the representation provided
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
        family_map: A dictionary or function that maps an RNA
            family name (e.g. '5s') to a vector (e.g. '[1, 0, 0]).
            If None, the file `family.np` is not written.
        verbosity (int): Verbosity level of the function. 1 (default)
            prints informative messages. 0 silences the function.
    """
    if verbosity:
        log.info("Format RNA data into Numpy files.")
    if dst[-1] != '/':
        dst += '/'
    # If the data is already encoded in the directory, exit.
    if _is_already_encoded(
            src, dst, max_size, primary_structure_map,
            secondary_structure_map, family_map):
        log.trace(f"The directory {dst} already contains the formatted data.")
        return
    # Create the directory if it des not exist.
    if not os.path.exists(dst):
        os.makedirs(dst)
    # Obtain the list of files to read.
    paths = []
    for path in pathlib.Path(src).rglob('*.ct'):
        paths.append(path)
    # Encode the content of each file.
    names = []           # RNA molecule names included in the dataset.
    rejected_names = []  # RNA molecule names excluded from the dataset.
    X = []               # Primary structure
    Y = []               # Secondary structure
    F = []               # Family
    for i, path in enumerate(paths):
        _, bases, pairings = rna_data.read_ct_file(str(path))
        family = diurnal.family.get_name(str(path))
        if len(bases) > max_size:
            rejected_names.append(str(path))
            continue
        names.append(str(path))
        X.append(primary_structure_map(bases, max_size))
        Y.append(secondary_structure_map(pairings, max_size))
        F.append(family_map(family))
        if verbosity:
            prefix = f"Encoding {len(paths)} files: "
            suffix = f" {path.name}"
            log.progress_bar(len(paths), i, prefix, suffix)
    if verbosity:
        print()  # Change the line after the progress bar.
        i = len(names)
        r = len(rejected_names)
        log.trace(f"Encoded {i} files. Rejected {r} files.")
    # Write the encoded file content into Numpy files.
    if not X:
        if verbosity:
            log.trace(f"No structure to write.")
        return
    s1 = dst + "primary_structures"
    if verbosity:
        log.trace(f"Writing primary structures at `{s1}.npy`.")
    np.save(s1, np.asarray(X, dtype=np.float32))
    s2 = dst + "secondary_structures"
    if verbosity:
        log.trace(f"Writing secondary structures at `{s2}.npy`.")
    np.save(s2, np.asarray(Y, dtype=np.float32))
    f = dst + "families"
    if verbosity:
        log.trace(f"Writing families at `{f}.npy`.")
    np.save(f, np.asarray(F, dtype=np.float32))
    n = dst + "names.txt"
    if verbosity:
        log.trace(f"Writing names at `{n}`.")
    with open(n, "w") as outfile:
        outfile.write("\n".join(names))
    # Write an informative file to sum up the content of the formatted folder.
    info = dst + "info.rst"
    if verbosity:
        log.trace(f"Writing an informative file at `{info}`.")
    with open(info, "w") as outfile:
        outfile.write(summarize(
            dst, primary_structure_map,
            secondary_structure_map, family_map))


def summarize(
        path: str,
        primary_structure_map,
        secondary_structure_map,
        family_map) -> str:
    """Summarize the content of the formatted file directory.

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
    content = "[> DIURNAL] RNA Database File Formatting\n"
    content += "========================================\n\n"
    content += f"Generation timestamp: {datetime.utcnow()} UTC\n\n"
    X = np.load(path + "primary_structures.npy")
    content += f"Number of structures: {X.shape[0]}\n\n\n"
    if X.any():
        content += "Primary Structure Encoding\n"
        content += "--------------------------\n\n"
        content += f"File: `{path + 'primary_structures.npy'}`\n\n"
        content += f"Shape: {X.shape}\n\n"
        example = "ACGU-"
        content += f"Encoding of the structure `{example}`:\n"
        code = primary_structure_map(example)
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
        example = [2, -1, 0]  # Corresponds to `(.)` in bracket notation.
        content += f"Encoding of the structure `{example}`:\n"
        code = secondary_structure_map(example)
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
        for f in diurnal.family.NAMES:
            name = family_map(f[0])
            content += f"    {f[0]} -> {name}"
            content += "\n"
        content += "\nExample: \n"
        content += str(F[0])
        content += "\n"
    return content
