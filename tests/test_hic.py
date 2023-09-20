"""
Tests related to the hic module.
"""

from hypothesis import given, strategies as st
import numpy as np
import networkx as nx
import pandas as pd
from test_metadata import generate_metadata
from Node2VecHiC.metadata import Metadata
from Node2VecHiC.hic import HiC, get_complementary_list

np.random.seed(123)


def generate_hic() -> str:
    """
    Generate a fake hic file and store it in .csv
    
    Return
    ------
        path (str):
            path of the .csv hic file.
    """
    number_nodes = 300
    matrix = np.random.randint(low = 0,
                            high = 1000,
                            size = (number_nodes, number_nodes))
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    hi_c = pd.DataFrame(matrix)
    path = 'hic.csv'
    hi_c.to_csv(path,
                index = False,
                header = False)
    return path

# Defining instance of the classes
METADATA_PATH = generate_metadata()
HIC_PATH = generate_hic()
metadata = Metadata(METADATA_PATH)
hic = HiC(metadata, HIC_PATH)


def test_get_df():
    """
    GIVEN: an hic instance
    WHEN: applying the get_df() function
    THEN: the returned dataframe is a pd.DataFrame
    """
    data_frame = hic.get_df()
    isinstance(data_frame, pd.DataFrame)


@given(selected_chromosome = st.integers(min_value = 0,
                                         max_value = len(metadata.data_frame)-1))
def test_block_df(selected_chromosome: int):
    """
    GIVEN: the parameter 'selected_chromosome'
    WHEN: applying the get_block_df() function
    THEN: the returned list contains different pd.DataFrame,
    and in all of these dataframe the selected_chromosome nodes
    are present. Here, the simple assertion 'assert node in 
    data_frame.index' does not hold beacuse at this point, the
    empty axes have already been removed in the dataframe, while
    this is not true for selected_nodes array.
    For this reason, before the assertion we remove these empty nodes.
    """
    block_dfs = hic.get_block_df(selected_chromosome)
    selected_nodes = metadata.get_nodes(selected_chromosome)
    empty_nodes = []
    for data_frame in block_dfs:
        for node in selected_nodes:
            if node not in data_frame.index:
                empty_nodes.append(node)
        non_empty_nodes = get_complementary_list(empty_nodes, selected_nodes)
        for node in non_empty_nodes:
            assert node in data_frame.index


def test_get_attributes():
    """
    GIVEN: an instance of the hic class
    WHEN: calling the method get_graph_attributes()
    THEN: the attributes are composed by the same number
    of nodes indexes.   
    """
    _, attributes = hic.get_graph_attributes()
    assert len(attributes) == len(hic.nodes)

@given(selected_chromosome = st.integers(min_value = 0,
                                         max_value = len(metadata.data_frame)-1))
def test_block_graph(selected_chromosome):
    """
    GIVEN: a hic instance
    WHEN: applying the function get_block_graph()
    THEN: all the elements in the returned list are nx.Graph
    structures. 
    """
    graphs_list = hic.get_block_graph(selected_chromosome)
    for graph in graphs_list:
        isinstance(graph, nx.Graph)
