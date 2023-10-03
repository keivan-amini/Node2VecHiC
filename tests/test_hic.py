"""
Tests related to the hic module.
"""

from hypothesis import given, strategies as st
import numpy as np
import networkx as nx
import pandas as pd
from test_metadata import generate_metadata
from Node2VecHiC.metadata import Metadata
from Node2VecHiC.hic import HiC, remove_empty_axis, get_complementary_list

np.random.seed(123)


def generate_hic() -> str:
    """
    Generate a fake hic file and store it in .csv
    
    Return
    ------
        path (str):
            path of the .csv hic file.
    """
    number_nodes = 8
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


def test_square_property_df():
    """
    GIVEN: an hic instance
    WHEN: applying the get_df() function
    THEN: the returned dataframe contains a squared matrix.
    """
    data_frame = hic.get_df()
    n_rows = data_frame.shape[0]
    n_columns = data_frame.shape[1]
    assert n_rows == n_columns

def test_symmetric_property_df():
    """
    GIVEN: an hic instance
    WHEN: applying the get_df() function
    THEN: the returned dataframe contains a symmetric matrix.
    """
    data_frame = hic.get_df()
    assert (data_frame == data_frame.T).all().all()


def test_diagonal_df():
    """
    GIVEN: an hic instance
    WHEN: applying the get_df() function
    THEN: the returned dataframe has just 0 values in the diagonal
    """
    data_frame = hic.get_df()
    assert np.all(np.diag(data_frame) == 0)

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

def test_get_complementary_list():
    """
    GIVEN: a reference list and a main list example
    WHEN: applying the function get_complementary_list()
    THEN: the output list is equal to the expected result.
    """
    reference_list = list(range(10))
    main_list = [0, 1, 3, 5, 7, 8]
    expected_list = [2, 4, 6, 9]
    output_list = get_complementary_list(main_list, reference_list)
    assert expected_list == output_list