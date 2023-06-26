"""
    Message logging module.

    Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    Affiliation: Département d'informatique, UQÀM
    File creation date: April 2023
    License: MIT
"""

import sys
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style


colorama_init()


def info(message: str) -> None:
    """Print information about the execution of the program.

    Args:
        message (str): Message to display.
    """
    print(f"[{Fore.GREEN}> DIURNAL{Style.RESET_ALL}] Info: {message}")


def trace(message: str) -> None:
    """Print a trace (i.e. pedantic) message.

    Args:
        message (str): Message to display.
    """
    print(f" {Fore.GREEN}>{Style.RESET_ALL} {message}")


def error(message: str) -> None:
    """Print an error message.

    Args:
        message (str): Message to display.
    """
    print(f"[{Fore.RED}> DIURNAL{Style.RESET_ALL}] Error: {message}")


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
    back = '\033[K\r'
    dash = f' {Fore.GREEN}>{Style.RESET_ALL} '
    sys.stdout.write(back + dash + prefix + bar + suffix)
    sys.stdout.flush()
