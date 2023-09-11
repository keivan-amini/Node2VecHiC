"""
TODO: This file must change position. At the moment, it's in scripts folder.
In the future should be moved in the tests folder.
To run tests with coverage go in the scripts folder and execute:
    python3.9 -m pytest --cov --cov-report term-missing

"""

from hypothesis import given, strategies as st
import pandas as pd

from metadata import Metadata

METADATA_PATH = '..\\data\\metadata_hic.xlsx'
metadata = Metadata(METADATA_PATH)


def test_get_df():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_df() function
    THEN: the returned dataframe is a pd.DataFrame
    """
    data_frame = metadata.get_df()
    isinstance(data_frame, pd.DataFrame)

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

def test_index_chromosomes():
    """
    GIVEN: a metadata instance
    WHEN: applying the get_index_chromosomes() function
    THEN: start and end columns share the same number
    of rows, different from zero.
    """
    start, end = metadata.get_index_chromosomes()
    assert len(start) == len(end) and len(start) != 0

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
    assert len(nodes) - 1 == metadata.end[chromosome] - metadata.start[chromosome]

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
