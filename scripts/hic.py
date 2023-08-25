"""
This script has the aim to construct the HiC class, useful to work
on an adjacency matrix and thereafter perform a statistical analysis.
It requires also a metadata file stored in csv or .xlsx containing in
the columns the name of the involved chromosomes, the start and the end 
of the nodes index in the newtork associated with the chromosomes.
"""
import networkx as nx
import pandas as pd
import numpy as np
#openpyxl needed


class HiC:

    """
    Description
    -----------
    This class creates and prepares dataframes structures,
    starting from an Hi-C .csv data interpreted as an adjacency
    matrix and a metadata file, containing informations about
    chromosomes.

    Methods
    -------
        get_dfs()
        remove_empty_axis()
        get_index_chromosomes()
        get_nodes()
        get_block_df()
        get_graph()
        get_block_graph()

    Attributes
    ----------
        name
        matrix_df
        metadata_df
        index_chromosomes
        attributes

    """

    def __init__(self, matrix_path: str,
                 metadata_path: str) -> None:

        """
        Remove empty axis and extract diagonal blocks, making
        the dataframe ready to undergo different algorithms.

        Parameters
        ----------
            matrix_path (str): 
                Path containing the adjacency matrix in .csv extension.
            metadata_path (str): 
                Path containing the metadata in .xlsx or .csv extensions.
        """

        self.name = matrix_path.replace(r'..\\data\\', '').replace('.csv', '')
        self.matrix_df, self.metadata_df = None, None

        self.get_dfs(matrix_path, metadata_path)
        self.get_index_chromosomes()

        self.get_graph()
        self.get_block_graph()

    def get_dfs(self, matrix_path: str,
                metadata_path: str) -> None:
        """
        Create two dataframes: one containing the adjacency 
        matrix and one containing the metadata. At the end,
        remove empty axis from the matrix_df attribute.
        
        Parameters
        ----------
            matrix_path (str):
                string containing the path of the adjacency matrix.
            metadata_path (str): 
                string containing the path of the metadata file.
        """
        self.matrix_df = pd.read_csv(matrix_path, header = None)
        if metadata_path.endswith(('xlsx',  'xls')):
            self.metadata_df = pd.DataFrame(pd.read_excel(metadata_path))
        elif metadata_path.endswith('csv'):
            self.metadata_df = pd.read_csv(metadata_path)
        else:
            raise ValueError(
                "Unsupported file format. Only Excel (xlsx, xls) and csv files are supported.")
        self.remove_empty_axis()

    def remove_empty_axis(self) -> None:
        """
        Remove empty axis from the dataframe containing the
        adjacency matrix coming from the Hi-C data.
        """
        self.matrix_df = self.matrix_df.loc[:, (self.matrix_df != 0).any(axis = 0)]
        self.matrix_df = self.matrix_df.loc[(self.matrix_df != 0).any(axis = 1)]

    def get_index_chromosomes(self) -> None:
        """
        Create a list containing the index referred to the
        starting node of each chromosome in the network.
        """
        self.index_chromosomes = self.metadata_df.end.values.tolist()
        self.index_chromosomes = np.append(self.index_chromosomes, 0, 0)

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
        -------
            nodes (np.ndarray):
                index nodes associated with the input
                chromosome.

        """
        nodes = np.arange(self.index_chromosomes[chromosome],
                          self.index_chromosomes[chromosome + 1])
        return nodes

    def get_block_df(self,
                     selected_chromosome: int = 1) -> list:
        """
        Create a list containing data frames corresponding to the
        selected chromosome and all the other chromosomes.

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
                From each df in block_dfs, the empty axes are removed.
                
        """
        nodes = np.arange(self.index_chromosomes[-1])
        chromosomes = np.arange(len(self.metadata_df.columns))
        block_dfs = []

        for chromosome in chromosomes:
            selected_chromosome_nodes = self.get_nodes(selected_chromosome)
            if chromosome != selected_chromosome:
                nodes_to_keep = self.get_nodes(chromosome)
                nodes_to_keep = np.append(nodes_to_keep, selected_chromosome_nodes)
                nodes_to_drop = get_complementary_list(nodes_to_keep, nodes)
                block_df = self.matrix_df.drop(index = nodes_to_drop,
                                               columns = nodes_to_drop)
                block_dfs.append(block_df)
        return block_dfs

    def get_graph(self): #TODO later
        """
        Get the graph structure of the Hi-C matrix using networkX
        library, starting from the matrix_df attribute.
        """
        graph = nx.from_pandas_adjacency(self.matrix_df)
        list_chromosomes = []
        for index in range(len(self.metadata_df.chr)):
            list_chromosomes += ([self.metadata_df.chr[index]] *
                     (self.metadata_df.end[index] - self.metadata_df.start[index] + 1))
        dict_chromosomes = {index: value for index, value in enumerate(list_chromosomes)}
        nx.set_node_attributes(graph, dict_chromosomes, 'chromosome')
        self.attributes = nx.get_node_attributes(graph, 'chromosome')

    def get_block_graph(self):
        """
        Get a list containing the graphs associated with the
        block data frames, for each couple of chromosomes in
        the metadata file.
        """
        block_dfs = self.get_block_df()
        graphs_list = []

        for block_df in block_dfs:
            block_graph = nx.from_pandas_adjacency(block_df)
            graphs_list.append(block_graph) # i think we do not need block attribute!

def get_complementary_list(main_list: list,
                           reference_list: list) -> list:
    """
    Get a list of elements from the reference list that are not present in the main list.

    Parameters
    ----------
        main_list: list
            List of elements to be used as the base of comparison.
        reference_list: list
            List of elements from which complementary elements are extracted.

    Return
    ------
        complementary_list: list
            list containing elements from the reference_list that are not present in the main_list.
    """
    complementary_list = [num for num in reference_list if num not in main_list]
    return complementary_list
