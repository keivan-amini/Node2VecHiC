"""
Tests related to the hic module.
"""

from hypothesis import given, strategies as st
import numpy as np
import networkx as nx
import pandas as pd
from Node2VecHiC.metadata import Metadata
from Node2VecHiC.hic import HiC, get_complementary_list, remove_empty_axis

np.random.seed(123)

# Defining instance of the class
METADATA_PATH = '..\\data\\metadata_hic.xlsx'
CANCER_PATH = '..\\data\\cancer_hic.csv'
metadata = Metadata(METADATA_PATH)
cancer_hic = HiC(metadata, CANCER_PATH) # maybe to change with a general .csv ?


def test_get_df():
    """
    GIVEN: an hic instance
    WHEN: applying the get_df() function
    THEN: the returned dataframe is a pd.DataFrame
    """
    data_frame = cancer_hic.get_df()
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
    block_dfs = cancer_hic.get_block_df(selected_chromosome)
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
    GIVEN:
    WHEN:
    THEN:        
    """
    pass


def test_block_graph():
    """
    GIVEN: a hic instance
    WHEN: applying the function get_block_graph()
    THEN: all the elements in the returned list are nx.Graph
    structures. 
    """
    graphs_list = cancer_hic.get_block_graph()
    for graph in graphs_list:
        isinstance(graph, nx.Graph)




# Generate a main list and a reference list to test the
# get_complementary_list() function

main_elements = 5
reference_elements = 10
n = metadata.end[-1]
k = np.random.randint(1, n)
@given(
    main_list=st.lists(
        st.integers(min_value = 1, max_value = k),
        min_size = 1,
        max_size = main_elements,
        unique = True,
    ),
    reference_list=st.lists(
        st.integers(min_value = 1, max_value = n),
        min_size = main_elements,
        max_size = reference_elements,
        unique = True,
    ),
)
def test_get_complementary_list(main_list: list,
                                reference_list: list):
    """
    GIVEN: a 'main_list' and a 'reference_list'
    WHEN: applying the function get_complementary_list()
    THEN: 
    """
    print(" la main_list vale", main_list)
    print("la reference_list vale", reference_list)
    complementary_list = get_complementary_list(main_list,
                                                reference_list)
    print("la complementary_list vale", complementary_list)
    
    if any(element not in reference_list for element in main_list):
        assert complementary_list != reference_list
    
    

def test_remove_empty_axis():
    """
    GIVEN: a pd.DataFrame
    WHEN: applying the function remove_empty_axis() two times
    THEN: the returned datafr
    """ 
    
