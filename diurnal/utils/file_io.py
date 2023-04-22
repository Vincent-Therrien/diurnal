"""
    File manipulation module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import sys
import tarfile
import requests

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

def read_ct_file(path: str) -> tuple:
    """
    Read a CT (Connect table) file and return its information.

    Args:
        path (str): File path of the CT file.

    Returns (tuple):
        The returned tuple contains the following data:
        - RNA molecule title.
        - Primary structure (i.e. a list of 'A', 'C', 'G', and 'U').
        - Pairings (i.e. a list of integers indicating the index of the
            paired based, with `-1` indicating unpaired bases).
    """
    bases = []
    pairings = []
    with open(path) as f:
        header = f.readline()
        if header[0] == ">":
            length = int(header.split()[2])
        else:
            length = int(header.split()[0])

        title = " ".join(header.split()[1:])
        f.seek(0)

        for i, line in enumerate(f):
            # deal w/ header for nth structure
            if i == 0:
                if header[0] == ">":
                    length = int(line.split()[2])
                else:
                    length = int(header.split()[0])

                title = " ".join(line.split()[1:])
                continue

            bn, b, _, _, p, _ = line.split()

            if int(bn) != i:
                raise NotImplementedError(
                    "Skipping indices in CT files is not supported."
                )

            bases.append(b)
            pairings.append(int(p) - 1)

            if i == length:
                break

    return title, bases, pairings
