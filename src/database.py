"""
    RNA secondary structure installation (i.e. download and unwrapping)
    module.
"""

import tarfile
import requests

# Constant values.
GITHUB_ROOT_URL = "https://raw.githubusercontent.com/"
DATABASE_URL = GITHUB_ROOT_URL + "Vincent-Therrien/rna-2s-database/main/"
ALLOWED_DATASETS = ["archiveII", "RNASTRalign"]
COMPRESSED_FILE_ENDING = ".tar.gz"

# Installation functions.
def download(storage_path: str, datasets: list(str), verbosity: int=1) -> None:
    """
    Download and unpack RNA secondary structure databases.

    This function downloads the datasets listed in the `datasets`
    argument, places them in the `storage_path` directory, and unpacks
    the downloaded files.

    Args:
        storage_path (str): Directory path in which the files are
            downloaded and unwrapped.
        datasets (list(str)): The list of databases to download. The
            allowed databases are `archiveII` and `RNASTRalign`.
        verbosity (int): Verbosity of the function. 1 (default) prints
            informative messages. 0 silences the function.
    """
    # Data validation.
    for dataset in datasets:
        if dataset not in ALLOWED_DATASETS:
            print(f"The dataset `{dataset}` is not allowed. ", end="")
            print(f"Allowed databases are {ALLOWED_DATASETS}.")
            raise FileNotFoundError
    # Data obtention.
    for dataset in datasets:
        # Download the file.
        url = DATABASE_URL + dataset + "/" + dataset + COMPRESSED_FILE_ENDING
        downloaded_file_name = storage_path + dataset + COMPRESSED_FILE_ENDING
        if verbosity:
            print(f"Downloading the file `{url}` at `{downloaded_file_name}`.")
        r = requests.get(url, allow_redirects=True)
        open(downloaded_file_name, 'wb').write(r.content)
        # Extract the file.
        if verbosity:
            print(f"Unwrapping the file `{downloaded_file_name}`.")
        tar = tarfile.open(downloaded_file_name, "r:gz")
        tar.extractall(downloaded_file_name[:-len(COMPRESSED_FILE_ENDING)])
        tar.close()
        if verbosity:
            print(f"Files installed in directory `{storage_path+'/'+dataset}`.")

def download_all(storage_path: str, verbosity: int=1) -> None:
    """
    Download all available RNA secondary structure databases.

    Args:
        storage_path (str): Directory path in which the files are
            downloaded and unwrapped.
        verbosity (int): Verbosity of the function. 1 (default) prints
            informative messages. 0 silences the function.
    """
    download(storage_path, ALLOWED_DATASETS, verbosity)
