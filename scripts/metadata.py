"""
This script has the aim to create the Metadata class, useful
to define Hi-C data characteristics interpreted as an adjacency
matrix of a network.
"""

import numpy as np
import pandas as pd


class Metadata:
    """
    Description
    -----------
    This class is useful to save informations regarding a metadata
    file containg biological informations related to Hi-C datas.
    The main prerequisite for calling this class is to have a table
    organised as the following example:

    |   chr |   start |   end |
    |-------|---------|-------|
    |  chr1 |       1 |   250 |
    |  chr6 |     251 |   422 |
    |   ... |     ... |   ... |

    Columns meaning:
    'chr' denotes the name of the chromosome;
    'start' denotes the starting index node of the chromosome;
    'end' denotes the ending index node of the chromosome.
    e.g, the nodes of the adjacency matrix between index 1 and
    250 are related to chromosome 1.
    The table must be stored in .csv or .xlsx, .xls.


    Methods
    -------
        get_df(path)
            given a path, return the dataframe of the metadata
            file. The function consider the cases in which
            the file is in different data format.
        get_index_chromosomes()
            return lists containing the starting index and the
            ending index for each chromosome, namely the column
            'start' and 'end' of the metadata dataframe.
        get_nodes(chromosome)
            given a input chromosome, return the index nodes
            of the network associated with the chromosome.
        get_dict_chromosomes():
            return a dict conatining indices as keys and name
            of the chromosomes as values.

    Attributes
    ----------
        path (str):
            path of the metadata file.
        data_frame (pd.DataFrame):
            dataframe of the metadata file.
        chromosomes (np.ndarray):
            numpy array defining the involved chromosome
            in the metadata file.
        start_chromosome (list):
            list containing the starting index node of each
            chromosome.
        end_chromosome (list):
            list containing the ending index node of each
            chromosome.
        attributes (dict):
            dict that associate each node index of the network
            to the name of the chromosome.
    """

    def __init__(self, path):
        """
        Initialize class by defining the attributes variables.
        
        Parameter
        ---------
            path (str): 
                string containing the metadata path in .xlsx, .xls
                or .csv extensions.
        """

        self.path = path
        self.data_frame = self.get_df()
        self.chromosomes = np.arange(len(self.data_frame))
        self.start, self.end = self.get_index_chromosomes()
        self.attributes = self.get_dict_chromosomes()

    def get_df(self) -> None:
        """
        Get the dataframe containing the metadata.

        Return
        ------
            data_frame (pd.DataFrame):
                dataframe of the metadata file.
        """
        if self.path.endswith(('xlsx',  'xls')):
            data_frame = pd.DataFrame(pd.read_excel(self.path))
        elif self.path.endswith('csv'):
            data_frame = pd.read_csv(self.path)
        else:
            raise ValueError(
                "Unsupported file format. Only Excel (xlsx, xls) and csv files are supported.")
        return data_frame

    def get_index_chromosomes(self) -> (list, list):
        """
        Return two list containing the index referred to the
        starting index and the ending index of each chromosome
        in the network.

        Returns
        -------
            start (list):
                list containing data in the second column of the
                dataframe, containing the starting index related
                to each chromosome.
            end (list):
                same of start variable, but now return the ending
                index related to each chromosome.
        """
        start, end = self.data_frame.start.values.tolist(), self.data_frame.end.values.tolist()
        return start, end

    def get_nodes(self, chromosome: int) -> np.ndarray:
        """
        Given a chromosome, return a np.ndarray containing
        the index nodes associated with the chromosome.

        Parameter
        ---------
            chromosome (int):
                integer value representing the chromosome
                we want to get the index nodes.

        Return
        ------
            nodes (np.ndarray):
                index nodes associated with the input
                chromosome.
        """
        nodes = np.arange(self.start[chromosome], self.end[chromosome])
        return nodes

    def get_dict_chromosomes(self) -> dict:
        """
        Return
        ------
            dict_chromosomes (dict):
                dict containing 
        """
        list_chromosomes = []
        for index in self.chromosomes:
            list_chromosomes += ([self.data_frame.chr[index]] * (self.data_frame.end[index] - self.data_frame.start[index] + 1))
        dict_chromosomes = dict(enumerate(list_chromosomes))
        return dict_chromosomes
