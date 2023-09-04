# Node2Vec on Hi-C: Detecting Chromosome Translocations with Network Analysis 🧬
## Introduction
In the realm of genomics, unraveling the intricacies of chromosome translocations holds paramount importance, particularly in the context of understanding genetic diseases, such as cancer. This repository introduces a simple and intuitive framework that combines graph theory and machine learning to detect translocation, given an Hi-C input data.

This repository is not designed as a genomic data analysis library. Instead, it relies on the [Node2Vec](https://github.com/eliorc/node2vec) and [scikit-learn](https://scikit-learn.org/stable/) libraries, saving you time on the data engineering part. Think of it as a tool tailored for visualizing genomic translocations when you have both a metadata file and Hi-C data at your disposal.

The repository is composed by different folders:
* [data](https://github.com/keivan-amini/Node2Vec-Hi-C/tree/main/data) contains Hi-C data and Metadata file.
* [scripts](https://github.com/keivan-amini/Node2Vec-Hi-C/tree/main/scripts) contains all python modules useed in the project.
* [model](https://github.com/keivan-amini/Node2Vec-Hi-C/tree/main/model) contains the outputs embeddings from the Node2Vec algorithm.
* [plot](https://github.com/keivan-amini/Node2Vec-Hi-C/tree/main/plot) contains the plotted figures.
* [tutorial](https://github.com/keivan-amini/Node2Vec-Hi-C/tree/main/tutorial) contains an easy how-to notebook exploiting this repository. TODO
* [tests](https://github.com/keivan-amini/Node2Vec-Hi-C/tree/main/tests) contains tests for the scripts. TODO
* [pdf](https://github.com/keivan-amini/Node2Vec-Hi-C/tree/main/pdf) contains the full project documentation, with mathematical theory, figures and results.


## Requirements
### Data requirements
* an Hi-C data in .csv that will be interpreted as an adjacency matrix of a graph:

      0      31495  12592  8465   5828   ...
      31495  0      55588  32895  21299  ...
      12592  55588  0      57642  32376  ...
      8465   32895  57642  0      42695  ...
      5828   21299  32376  42695  0      ...
      ...    ...    ...    ...    ...    ...


* a Metadata file in .csv or in .xlsx, .xls that gives informations about nodes of the graph and the related chromosome:

    |   chr |   start |   end |
    |-------|---------|-------|
    |  chr1 |       1 |   250 |
    |  chr6 |     251 |   422 |
    |   ... |     ... |   ... |

### Required libraries
See the [requirements.txt](https://github.com/keivan-amini/Node2Vec-Hi-C/blob/main/requirements.txt) file.

## Installation
Open a terminal or command prompt. Navigate to your preferred directory where you want to clone the repository and run:
```
git clone https://github.com/keivan-amini/Node2Vec-Hi-C
```
Move to the cloned directory and install the required libraries:
```
cd Node2Vec-Hi-C
python3 -m pip install -r requirements.txt
```
## Tutorials
TODO
## Tests
TODO
## Plot examples
TODO
## References and thanks
This repository contains the project for the course of Complex Networks, part of the MSc in Applied Physics at the University of Bologna. All the references for the project are contained in the  `.pdf ` document. Special thanks to [Daniel Remondini](https://www.unibo.it/sitoweb/daniel.remondini) and [Alessandra Merlotti](https://www.unibo.it/sitoweb/alessandra.merlotti2)!
