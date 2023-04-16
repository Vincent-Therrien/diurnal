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
            diurnal.database.summarize("./data/")
            diurnal.database.format(
                "./data/", # Directory of the raw data to format.
                s1.IUPAC_ONEHOT, # RNA encoding scheme.
                s2.DOT_BRACKET_ONEHOT, # RNA encoding scheme.
                "./data/formatted" # Formatted data output directory.
            )
            diurnal.database.summarize("./data/formatted")
            diurnal.database.visualize("./data/formatted")

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import shutil
import os
import pathlib

from .utils import file_io

# Constant values.
URL_PREFIX ="https://github.com/Vincent-Therrien/rna-2s-database/raw/main/data/"
FILE_ENDING = ".tar.gz"
ALLOWED_DATASETS = ["archiveII", "RNASTRalign"]

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
    if verbosity: print("[> DIURNAL]: Download and install an RNA database.")
    if dst[-1] != '/':
        dst += '/'
    # Data validation.
    if type(datasets) is str:
        datasets = [datasets]
    for dataset in datasets:
        if dataset not in ALLOWED_DATASETS:
            print(f"    The dataset `{dataset}` is not allowed. ", end="")
            print(f"Allowed databases are {ALLOWED_DATASETS}.")
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
            print(f"    Files installed in the directory `{dst + dataset}`.")
            print()

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

def format_dataset(database_path: str,
                   primary_structure_format,
                   secondary_structure_format,
                   formatted_path: str,
                   max_size: int,
                   verbosity: int=1) -> None:
    """
    Transform the original datasets into the representation provided by
    the arguments.

    This function reads the RNA dataset files comprised in the directory
    `dataset_path`, applies the encoding schemes defined by the
    arguments, and writes the result in the `formatted_path` directory.

    Args:
        database_path (str): The directory in which RNA datasets are
            located. If you used `./data/` as the storage path of the
            `download` function, then the directory `./data/` must be
            provided to this function.
        primary_structure_format: A dictionary or function that maps an
            RNA primary structure symbol ('A', 'C', 'G', 'U') to a
            numeric value (e.g. map A to [1, 0, 0, 0]).
        secondary_structure_format: A dictionary or function that maps
            an RNA secondary structure symbol ('(', '.', ')') to a
            numeric value (e.g. map '.' to [0, 1, 0]).
        formatted_path (str): The directory in which the encoded RNA
            structures are written.
        max_size (int): Maximal number of nucleotides in an RNA
            structure. If an RNA structure has more bases than this
            number, it is not included in the formatted dataset.
        verbosity (int): Verbosity of the function. 1 (default) prints
            informative messages. 0 silences the function.
    """
    pass

def summarize(path: str):
    """
    """
    pass

def visualize(path: str):
    """
    """
    pass
