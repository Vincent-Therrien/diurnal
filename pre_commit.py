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


if os.name == 'nt':
    r = subprocess.run(["py", "setup.py", "install"], capture_output=True)
else:
    r = subprocess.run(["python3", "setup.py", "install"], capture_output=True)

log.trace("Building the documentation.")
doc_failed = False
path = os.path.join('docs', 'make')
r = subprocess.run([path, "clean"], capture_output=True, shell=True)
r = subprocess.run(
    ["sphinx-apidoc", "-o", "./docs/source", "./diurnal"],
    capture_output=True, text=True)
if r.returncode:
    log.error("Cannot generate documentation from the source code.")
    doc_failed = True
path = os.path.join('docs', 'make')
r = subprocess.run([path, "html"], capture_output=True, shell=True)
if r.returncode:
    log.error("Cannot build the documentation.")
    doc_failed = True
if not doc_failed:
    log.info("Successfully updated and built the documentation.")

log.trace("Source code analysis with `pycodestyle`.")
directories = ["diurnal", "test", "demo"]
n_lines = {}
n_style_errors = 0
for directory in directories:
    n_lines[directory] = 0
    for path in pathlib.Path(directory).rglob('*.py'):
        r = subprocess.run(
            ["pycodestyle", str(path)], capture_output=True, text=True)
        n = len(r.stdout.split('\n')) - 1
        if n:
            print(r.stdout[:-1])
        n_style_errors += n
        with open(str(path), 'r') as file:
            n_lines[directory] += file.read().count('\n')
log.trace(f"Analyzed {sum(n_lines.values())} lines of code.")
log.trace(f"The diurnal library comprises {n_lines['diurnal']} lines of code.")
if n_style_errors:
    log.error(f"Detected {n_style_errors} style errors.")
else:
    log.info("Detected no style errors.")

log.trace("Test execution.")
r = subprocess.run(
    ["pytest", "test", "--tb=short"], capture_output=True, text=True)
print(r.stdout)
tests_failed = True if r.returncode != 0 else False
if tests_failed:
    log.error("Test execution failed.")
else:
    log.info("Test execution succeeded.")

if n_style_errors > 0 or tests_failed:
    log.error("The commit cannot be integrated to the project.")
    sys.exit(1)
log.info("Success: The commit can be integrated to the project.")
