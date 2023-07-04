"""
    Validation script used to ensure that the repository follows the
    requirements of the project.

    To execute it, run the command ``py pre_commit.py``.
"""

import os
import pathlib
import subprocess
import sys

from diurnal.utils import log


os.chdir(os.path.dirname(os.path.abspath(__file__)))


log.trace("Source code analysis with `pycodestyle`.")
directories = ["diurnal", "test", "demo"]
n_style_errors = 0
for directory in directories:
    for path in pathlib.Path(directory).rglob('*.py'):
        r = subprocess.run(
            ["pycodestyle", str(path)], capture_output=True, text=True)
        n = len(r.stdout.split('\n')) - 1
        if n:
            print(r.stdout[:-1])
        n_style_errors += n
if n_style_errors:
    log.error(f"Detected {n_style_errors} style errors.")
else:
    log.info("Detected no style errors.")

log.trace("Test execution.")
r = subprocess.run(["pytest", "test"], capture_output=True, text=True)
print(r.stdout)
failed = True if r.returncode != 0 else False
if failed:
    log.error("Test execution failed.")
else:
    log.info("Test execution succeeded.")

if n_style_errors > 0 or failed:
    log.error("The commit cannot be integrated to the project.")
    sys.exit(1)
log.info("Success: The commit can be integrated to the project.")
