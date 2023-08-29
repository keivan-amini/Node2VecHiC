"""
This script has the aim to construct the HiC class, useful to work
on an adjacency matrix and thereafter apply algorithms on the obtained
network. 
It is essential to have a metadata file containing useful informations
regarding the Hi-C matrices; this can be give as input to the Metadata
class defined in the metadata.py module.
"""

import networkx as nx
import pandas as pd
import numpy as np
#openpyxl needed


class HiC:

    """
    Description
    -----------
    This class creates and prepares a dataframe structure
    starting from an Hi-C data, interpreted as an adjacency
    matrix of a network. The matrix data must be stored in
    a .csv file.

    The class takes in input two parameters: a metadata object,
    that can be defined with using the Metadata() class contained
    in the metadata.py script, and the path string related to the
    stored adjacency matrix.
    

    Methods TODO 
    -------
        get_df()
            transform .csv adjacency matrix
            in Pandas dataframe format.
        get_block_df(selected_chromosome = 1)
            return a list containing dataframes for all possible
            pair of chromosomes with the input selected_chromosome.
        get_attributes()
            get the attributes of the adjacency matrix structure.
        get_block_graph()
            return a list containing graphs related to the block_df
            list. #TODO maybe change


    Attributes TODO
    ----------
        metadata (object):
            instance of the class Metadata().
        chromosomes (np.ndarray):
            attribute of the Metadata() class, containing the
            involved chromosome in the metadata file.
        nodes (np.ndarray):
            attribute of the Metadata() class, containing the
            indexes of involved nodes in the metadata.
        path (string):
            path of the adjacency matrix. The matrix must be
            in .csv extension.
        name (string):
            name of the instance Hi-C class, extracted from
            the attribute path. TODO call it as the instance name!
        data_frame (pd.DataFrame):
            dataframe containing the main adjacency matrix.
        attributes (dict):
            dictionary containing the index nodes related to
            each chromosome.

    """

    def __init__(self, metadata: object,
                 matrix_path: str) -> None:

        """
        Initialize class by defining the attributes variables.

        Parameters
        ----------
            metadata (class):
                Metadata class previously defined containing informations
                about the index nodes of the network. The definition of the
                class is contained in the metadata.py module.
            matrix_path (str): 
                Path containing the adjacency matrix in .csv extension.
        """
        self.metadata = metadata
        self.chromosomes = self.metadata.chromosomes
        self.nodes = self.metadata.nodes

        self.path = matrix_path
        self.name = self.path.replace('..\\data\\', '').replace('.csv', '')
        self.data_frame = self.get_df()
        self.attributes = self.get_attributes()


    def get_df(self) -> pd.DataFrame:
        """
        Get the dataframe containing the Hi-C data interpreted
        as an adjacency matrix, with empty axis removed.
        
        Return
        ------
            data_frame (pd.Pandas.Dataframe):
                dataframe containing the adjacency matrix, with
                empty axis removed.
        """
        data_frame = pd.read_csv(self.path, header = None)
        data_frame = remove_empty_axis(data_frame)
        return data_frame

    def get_block_df(self,
                     selected_chromosome: int = 1) -> list:
        """
        Create a list containing dataframes holding only the
        nodes associated with the selected_chromosome and, in turn,
        all the other chromosomes. Each dataframes in question will
        thus consider just two chromosomes.

        Parameter
        ---------
            selected_chromosome (int), optional:
                Integer number between 0 and len(self.metadata_df), representing
                the selected chromosome to which we are going to create different
                block data frames, regarding all the possible couples between the
                selected chromosome and all the other chromosomes.
                Default is 1 (chr6), the one with unobservable traslocation.
        Return
        ------
            block_dfs (list):
                List containing the data frames considering only the
                selected chromosome and all the other chromosomes.
                
        """
        block_dfs = []
        for chromosome in self.chromosomes:
            selected_chromosome_nodes = self.metadata.get_nodes(selected_chromosome)
            if chromosome != selected_chromosome:
                nodes_to_keep = self.metadata.get_nodes(chromosome)
                nodes_to_keep = np.append(nodes_to_keep, selected_chromosome_nodes)
                nodes_to_drop = get_complementary_list(nodes_to_keep, self.nodes)
                block_df = self.data_frame.drop(index = nodes_to_drop,
                                        columns = nodes_to_drop,
                                        errors = 'ignore')
                block_dfs.append(block_df)
        return block_dfs

    def get_attributes(self) -> dict:
        """
        Get the attributes of the Hi-C matrix using networkX
        library, starting from the matrix data_frame.
        Note that dict_chromosomes and self.attributes are two
        different dictionaries because the matrix dataframe is
        in part incomplete. (to check) TODO

        Return
        ------
            attributes (dict):
                dictionary of attributes keyed by node related
                to the network structure of the adjacency matrix.
        """
        graph = nx.from_pandas_adjacency(self.data_frame)
        dict_chromosomes = self.metadata.get_dict_chromosomes()
        nx.set_node_attributes(graph, dict_chromosomes, 'chromosome')
        attributes = nx.get_node_attributes(graph, 'chromosome') #self.attributes is shorter! 749 nodes VS dict_chromosomes with 778.
        return attributes

    def get_block_graph(self) -> list: #TODO is it useful? #da capire se dargli in input una lista di blocchi a caso. (block_df instead of self.block_df)
        """
        Get a list containing the graphs associated with the
        block data frames, for each couple of chromosomes in
        the metadata file.

        Return
        ------
            graphs_list (list):
                list containing graphs structure related to the

        """
        block_dfs = self.get_block_df() # non ha senso questa linea
        graphs_list = []

        for block_df in block_dfs:
            block_graph = nx.from_pandas_adjacency(block_df)
            graphs_list.append(block_graph) # i think we do not need block attribute!
        
        return graphs_list

def get_complementary_list(main_list: list,
                           reference_list: list) -> list:
    """
    Get a list of elements from the reference list that are not present in the main list.

    Parameters
    ----------
        main_list (list):
            List of elements to be used as the base of comparison.
        reference_list (list):
            List of elements from which complementary elements are extracted.

    Return
    ------
        complementary_list (list):
            list containing elements from the reference_list that are not present in the main_list.
    """
    complementary_list = [num for num in reference_list if num not in main_list]
    return complementary_list


def remove_empty_axis(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Given an input pd.DataFrame, remove empty columns and rows.

    Parameter
    --------
        data_frame (pd.DataFrame):
            input dataframe we want to remove empty axes.
            
    Return
    ------
        cleaned_df (pd.DataFrame):
            output dataframe with empty axes removed.
    """
    cleaned_df = data_frame.loc[:, (data_frame != 0).any(axis = 0)]
    cleaned_df = cleaned_df.loc[(cleaned_df != 0).any(axis = 1)]
    return cleaned_df
