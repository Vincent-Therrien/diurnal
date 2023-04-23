#!/usr/bin/env python

import os
from distutils.core import setup

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
dependencies = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        dependencies = f.read().splitlines()

setup(name='diurnal',
    version='1.0',
    description='RNA secondary prediction utility library',
    author='Vincent Therrien',
    author_email='therrien.vincent.2@courrier.uqam.ca',
    url='https://github.com/Vincent-Therrien/diurnal',
    packages=['diurnal', 'diurnal/utils', 'diurnal/networks'],
    python_requires='>=3.7',
    install_requires=dependencies
)
