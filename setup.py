#!/usr/bin/env python

from distutils.core import setup

setup(name='diurnal',
    version='1.0',
    description='RNA secondary prediction utility library',
    author='Vincent Therrien',
    author_email='therrien.vincent.2@courrier.uqam.ca',
    url='https://github.com/Vincent-Therrien/diurnal',
    packages=['diurnal', 'diurnal/utils', 'diurnal/networks'],
    python_requires='>=3.7'
)
