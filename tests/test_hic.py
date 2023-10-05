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

def generate_block_df() -> list:
    """
    Manually generate a list of block data frame, for testing
    purposes on the function get_block_df(). In this context,
    selected_chromosome = 2, i.e., chr3.

    Return
    ------
        expected_block_dfs (list):
            list containing the expected dataframes using the
            function get_block_df() with parameter
            selected_chromosome = 2.
    """

    # chr1-chr3
    data1 = {1: [0, 227, 350],
             6: [227, 0, 433],
             7: [350, 433, 0]}

    # chr2-chr3
    data2 = {3: [0, 474, 140, 858],
             4: [474, 0, 613, 332],
             6: [140, 613, 0, 433],
             7: [858, 332, 433, 0]}

    df1 = pd.DataFrame(data1)
    df1.index = [1, 6, 7]

    df2 = pd.DataFrame(data2)
    df2.index = [3, 4, 6, 7]

    expected_block_dfs = [df1, df2]
    return expected_block_dfs

def generate_graph_list():
    """
    Manually generate a graph list in order to test the function
    get_block_graph(), with the input parameter selected_chromosome
    equals 1, for testing purposes.

    Return
    ------
        expected_graphs (list):
            expected graphs list by considering the parameter
            selected_chromosome = 1 in the get_block_graph()
            function.
    """
    # First graph
    nodes1 = [7, 3, 4, 1]
    edges1 = [(7, 3), (7, 4), (7, 1), (3, 4), (3, 1), (4, 1)]
    graph1 = nx.Graph()
    graph1.add_nodes_from(nodes1)
    graph1.add_edges_from(edges1)

    # Second graph
    nodes2 = [7, 6, 4, 3]
    edges2 = [(7, 6), (7, 4), (7, 3), (6, 4), (6, 3), (4, 3)]
    graph2 = nx.Graph()
    graph2.add_nodes_from(nodes2)
    graph2.add_edges_from(edges2)

    expected_graphs = [graph1, graph2]
    return expected_graphs


def generate_main_list():
    """
    Manually generate main_list in order to test the function
    get_complementary_list(). Default dim is random, but less
    than 10.

    Return
    ------
        main_list (list):
            list with dimension less or equal with respect to
            reference_list.
    """
    dim = np.random.randint(0, 10)
    main_list = set()
    while len(main_list) <= dim:
        main_list.add(np.random.randint(0, 10))
    return main_list


# Defining instance of the classes for testing purposes
METADATA_PATH = generate_metadata()
HIC_PATH = generate_hic()
metadata = Metadata(METADATA_PATH)
hic = HiC(metadata, HIC_PATH)




########## TEST FUNCTIONS ##########

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
    THEN: the returned dataframe has just 0 values in the diagonal.
    """
    data_frame = hic.get_df()
    assert np.all(np.diag(data_frame) == 0)


@given(selected_chromosome = st.integers(min_value = 0,
                                         max_value = len(metadata.data_frame)-1))
def test_nodes_block_df(selected_chromosome: int):
    """
    GIVEN: the parameter 'selected_chromosome'
    WHEN: applying the get_block_df() function
    THEN: in all the dataframes, the node associated with
    the selected chromosome are present.
    """
    block_dfs = hic.get_block_df(selected_chromosome)
    selected_nodes = metadata.get_nodes(selected_chromosome)
    for data_frame in block_dfs:
        for node in selected_nodes:
            assert node in data_frame.index

def test_get_block_df():
    """
    GIVEN: an hic, metadatata, and selected_chromosome parameter
    WHEN: applying the get_block_df() function
    THEN: the ouput list is equal to the expected result.
    """
    output_list = hic.get_block_df(selected_chromosome = 2)
    expected_list = generate_block_df()
    for output_df, expected_df in zip(output_list, expected_list):
        assert output_df.equals(expected_df)


def test_number_nodes_attributes():
    """
    GIVEN: an instance of the hic class
    WHEN: calling the method get_graph_attributes()
    THEN: the attributes are composed by the same number
    of nodes indexes.   
    """
    _, attributes = hic.get_graph_attributes()
    assert len(attributes) == len(hic.nodes)

def test_function_get_attributes():
    """
    GIVEN: an instance of the hic class and an expected dict
    WHEN: calling the method get_graph_attributes()
    THEN: the output dictionary is equal to the expected dictionary.
    """
    expected_dict = {0: 'chr1',
                     1: 'chr1',
                     2: 'chr2',
                     3: 'chr2',
                     4: 'chr2',
                     5: 'chr3',
                     6: 'chr3'}
    _, output_dict = hic.get_graph_attributes()
    assert output_dict == expected_dict

@given(selected_chromosome = st.integers(min_value = 0,
                                         max_value = len(metadata.data_frame)-1))
def test_len_blocks_graphs(selected_chromosome):
    """
    GIVEN: a hic instance and a selected_chromosome
    WHEN: applying the function get_block_graph()
    THEN: the output list graphs and the output list
    data frames share the same number of elements
    """
    graphs_list = hic.get_block_graph(selected_chromosome)
    dfs_list = hic.get_block_df(selected_chromosome)
    assert len(graphs_list) == len(dfs_list)

def test_function_get_block_graph():
    """
    GIVEN: a hic instance and an example selected chromosome = 1
    WHEN: applying the function get_block_graph()
    THEN: the output list graphs is equal to the expected
    result.
    """
    output_graphs_list = hic.get_block_graph(selected_chromosome = 1)
    expected_graph_list = generate_graph_list()
    for output_graph, expected_graph in zip(output_graphs_list, expected_graph_list):
        nx.is_isomorphic(output_graph, expected_graph)


def test_remove_empty_axis_square():
    """
    GIVEN: a data frame example without empty axis
    WHEN: applying the function remove_empty_axis()
    THEN: the output data frame is equal is the input
    data frame, i.e., the function did not do anything.
    """
    input_df = hic.data_frame
    output_df = remove_empty_axis(input_df)
    assert input_df.equals(output_df)

def test_column_remove_empty_axis():
    """
    GIVEN: a data frame example with a null column
    WHEN: applying the function remove_empty_axis()
    THEN: the output data frame is equal to the expected result.
    """
    data = {'A': [0, 2, 3],
            'B': [0, 1, 2],
            'C': [0, 5, 6]}
    data_frame = pd.DataFrame(data)
    output_data_frame = remove_empty_axis(data_frame)
    expected_result = {'A': [2, 3],
                       'B': [1, 2],
                       'C': [5, 6]}
    expected_data_frame = pd.DataFrame(expected_result)
    expected_data_frame.index = [1, 2]
    assert output_data_frame.equals(expected_data_frame)

def test_row_remove_empty_axis():
    """
    GIVEN: a data frame example with a null row
    WHEN: applying the function remove_empty_axis()
    THEN: the output data frame is equal to the expected result.
    """
    data = {'A': [1, 2, 3],
            'B': [0, 0, 0],
            'C': [4, 5, 6]}
    data_frame = pd.DataFrame(data)
    output_data_frame = remove_empty_axis(data_frame)
    expected_result = {'A': [1, 2, 3],
                       'C': [4, 5, 6]}
    expected_data_frame = pd.DataFrame(expected_result)
    assert output_data_frame.equals(expected_data_frame)

def test_n_elements_complementary_list():
    """
    GIVEN: a reference list and a random main list example
    WHEN: applying the function get_complementary_list()
    THEN: the len of the output list is equal the len of
    the reference list, minus the len of the main list.
    """
    reference_list = list(range(10))
    main_list = generate_main_list()
    output_list = get_complementary_list(main_list, reference_list)
    assert len(output_list) == len(reference_list) - len(main_list)

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
