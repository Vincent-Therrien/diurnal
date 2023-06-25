"""
    File manipulation module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import os
import sys
import tarfile
import requests
import pathlib


def log(message: str, level: int = 0) -> None:
    """
    Print information about the execution of the program.

    Args:
        message (str): Message to display.
        level (int): Logging level. `0` is a general message. `1` is a
            nested message. `-1` is an error message.
    """
    if level == 0:
        print(f"[> DIURNAL] Info: {message}")
    elif level == -1:
        print(f"[> DIURNAL] Error: {message}")
    else:
        print(f"    {message}")


def progress_bar(N: int, n: int, prefix: str="", suffix: str="") -> None:
    """
    Print a progress bar in the standard output.

    Args:
        N (int): Total number of elements to process.
        n (int): Number of elements that have been processed.
        prefix (str): A text to display before the progress bar.
        suffix (str): A text to display after the progress bar.
    """
    if n == N - 1:
        done = 50
    else:
        done = int(50 * n / N)
    bar = f"[{'=' * done}{' ' * (50-done)}]"
    sys.stdout.write('\033[K\r' + prefix + bar + suffix)
    sys.stdout.flush()


def clean_dir_path(directory: str) -> str:
    """Validate a directory.

    Args:
        directory (str): Directory name to validate.

    Raises:
        RuntimeError: If the directory does not exist.

    Returns (str): Cleaned directory path.
    """
    if not directory.endswith("/"):
        directory += "/"
    if not os.path.isdir(directory):
        log(f"Invalid directory: {directory}", -1)
        raise RuntimeError
    return directory


def download(url: str, dst: str, verbosity: int, name: str="") -> None:
    """
    Download a file through HTTPS.

    Args:
        url (str): Location of the file to download.
        dst (str): File path of the downloaded content.
        verbosity (int): Verbosity level of the function. `0` silences
            the function. `1` prints a loading bar.
        name (str): Name of the downloaded file - used for logging.
    """
    if verbosity > 1:
        print(f"Downloading the file `{url}` in `{dst}`.")
    with open(dst, "wb") as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if verbosity:
                    progress_bar(total_length, dl, f"    Downloading   {name} ")
    if verbosity:
        print()


def decompress(filename: str, mode: str, dst: str,
               verbosity: int, name: str="") -> None:
    """
    Decompress a TAR file.

    Args:
        filename (str): Name of the file to decompress.
        mode (str): Decompression mode (e.g. `r:gz`).
        dst (str): Output directory.
        verbosity (int): Verbosity level. `0` silences the function.
        name (str): Decompressed file name - used for logging.
    """
    tar = tarfile.open(filename, mode=mode)
    if verbosity:
        members = tar.getmembers()
        for i, member in enumerate(members):
            tar.extract(member, path=dst)
            if verbosity:
                progress_bar(len(members), i, f"    Decompressing {name} ")
    else:
        tar.extractall(dst)
    tar.close()
    if verbosity:
        print()


def is_downloaded(dst: str, n: int) -> bool:
    """Check if a dataset has been downloaded and is available on the
    filesystem.

    Args:
        dst (str): Expected directory in which the dataset should be.
        n (int): Expected number of RNA structure files.

    Returns (bool): True if the dataset is downloaded, False otherwise.
    """
    paths = []
    for path in pathlib.Path(dst).rglob('*.ct'):
        paths.append(path)
    return len(paths) == n
