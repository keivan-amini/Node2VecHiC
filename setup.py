from setuptools import setup, find_packages

setup(
    name = 'Node2VecHiC',
    version = '0.1.0',
    packages = find_packages(include=['Node2VecHiC']),
    description='A small package for applying node2vec algorithms to Hi-C matrices and visualize the results.',
    author='Keivan Amini',
    install_requires = ["networkx", "pandas", "matplotlib", "numpy", "scikit-learn", "node2vec",],
    )
