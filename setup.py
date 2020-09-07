# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="dvae-speech",
    version="1.0",
    author="Xiaoyu BIE",
    author_email="xiaoyu.bie@inria.fr",
    description="A PyTorch implementation of DVAE models on speech processing",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.inria.fr/xbie/dvae-speech',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'soundfile',
        'librosa',
        'torch>=1.3.0+cu92',
        'speechmetrics @ git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu]',
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)