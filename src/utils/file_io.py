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
import os

def download(url: str, dst: str, verbosity: int, name: str="") -> None:
    """
    Download a file through HTTPS.

    Args:
        url (str): Location of the file to download.
        dst (str): File path of the downloaded content.
        verbosity (int): Verbosity level of the function. `0` silences
            the function. `1` prints a loading bar.
        name (str): Name of the downloaded file.
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
                    done = int(50 * dl / total_length)
                    prefix = f"    Downloading   {name} "
                    bar = f"[{'=' * done}{' ' * (50-done)}]"
                    sys.stdout.write('\r' + prefix + bar)
                    sys.stdout.flush()
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
        name (str): 
    """
    tar = tarfile.open(filename, mode=mode)
    if verbosity:
        members = tar.getmembers()
        for i, member in enumerate(members):
            tar.extract(member, path=dst)
            prefix = f"    Decompressing {name} "
            if i == len(members) - 1:
                done = 50
            else:
                done = int(50 * i / len(members))
            bar = f"[{'=' * done}{' ' * (50-done)}]"
            sys.stdout.write('\r' + prefix + bar)
            sys.stdout.flush()
        print()
    else:
        tar.extractall(dst)
    tar.close()
