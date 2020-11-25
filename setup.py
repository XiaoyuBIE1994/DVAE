# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="dvae-speech",
    version="1.0",
    author="Xiaoyu BIE",
    author_email="xiaoyu.bie@inria.fr",
    description="Dynamical Variation Auto-Encoder for speech re-synthesis",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.inria.fr/xbie/dvae-speech',
    packages=find_packages(),
    # package_dir={'dvae', 'dvae'},
    install_requires=[
        'numpy',
        'matplotlib',
        'soundfile',
        'librosa',
        'torch>=1.3.0+cu92'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)