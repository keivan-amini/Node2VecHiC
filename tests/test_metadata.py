"""
Tests related to the metadata module.
"""

from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from Node2VecHiC.metadata import Metadata


def generate_metadata() -> str:
    """
    Generate a fake metadata file and store it in a .csv

    Return
    ------
        path (str):
            path of the .csv metadata file.

    """
    data = {'chr': ['chr1', 'chr2', 'chr3'],
            'start': [1, 3, 6],
            'end': [2, 5, 7]} 
    metadata = pd.DataFrame(data)
    path = 'metadata.csv'
    metadata.to_csv('metadata.csv', index = False)
    return path

# Defining instance of the class
METADATA_PATH = generate_metadata()
metadata = Metadata(METADATA_PATH)


def test_columns_name_df():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_df() function
    THEN: the structure of the returned dataframe is composed by
    columns indicating the chromosome, the start of the index
    node and the end of the index node.
    """
    accepted_columns = ['chr', 'start', 'end']
    data_frame = metadata.get_df()
    assert data_frame.columns.to_list() == accepted_columns

def test_number_rows_df():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_df() function
    THEN: all the columns of the returned dataframe share
    the same number of rows.
    """
    data_frame = metadata.get_df()
    columns_length = data_frame.apply(len)
    assert columns_length.nunique() == 1

def test_nodes_df():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_df() function
    THEN: all the nodes inside the metadata df belong
    to a certain chromosome.
    """
    data_frame = metadata.get_df()
    all_nodes = np.arange(metadata.start[0], metadata.end[-1]+1)
    nodes = []
    for _, row in data_frame.iterrows():
        nodes.extend(np.arange(row['start'], row['end'] + 1))
    assert np.array_equal(all_nodes, nodes)

def test_index_chromosomes():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_index_chromosomes() function
    THEN: start and end columns share the same number
    of rows, different from zero.
    """
    start, end = metadata.get_index_chromosomes()
    assert len(start) == len(end) and len(start) != 0

def test_function_index_chromosomes():
    """
    GIVEN: a metadata example instance
    WHEN: applying the get_index_chromosomes() function
    THEN: start and end lists are equal to the expected
    results, i.e., the function get_index_chromosomes()
    work properly.
    """
    expected_start = [1, 3, 6]
    expected_end = [2, 5, 7]
    start, end = metadata.get_index_chromosomes()
    assert (expected_start, expected_end) == (start, end)

@given(chromosome = st.integers(min_value = 0,
                              max_value = len(metadata.data_frame)-1))
def test_parameter_get_nodes(chromosome: int):
    """
    GIVEN: the parameter 'chromosome'
    WHEN: applying the get_nodes() function
    THEN: the returned list of nodes is composed by the right
    amount of elements, i.e. the difference between the end
    and the start columns (fixing a row) in the metadata.
    """
    nodes = metadata.get_nodes(chromosome)
    assert len(nodes) == metadata.end[chromosome] - metadata.start[chromosome]

def test_function_get_nodes():
    """
    GIVEN: a metadata instance, and a chromosome example
    WHEN: applying the get_nodes() function
    THEN: output nodes are equal to the expected results.
    """
    chromosome_example = 1
    expected_nodes = np.arange(3,5)
    nodes = metadata.get_nodes(chromosome_example)
    assert np.array_equal(nodes, expected_nodes)

def test_dict_chromosomes():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_dict_chromosomes() function
    THEN: the returned dict contains informations regarding
    all the nodes and all the chromosomes involved in the metadata.
    """
    dict_chr = metadata.get_dict_chromosomes()
    unique_keys = list(set(dict_chr.keys()))
    unique_values = list(set(dict_chr.values()))

    assert len(unique_keys) == metadata.end[-1]
    assert len(unique_values) == len(metadata.data_frame.chr)

def test_function_dict_chromosomes():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_dict_chromosomes() function
    THEN: output dict is equal to the expected result.
    """
    expected_dict = {0: 'chr1', 1: 'chr1',
                     2: 'chr2', 3: 'chr2', 4: 'chr2',
                     5: 'chr3', 6: 'chr3'}
    dict_chr = metadata.get_dict_chromosomes()
    assert dict_chr == expected_dict