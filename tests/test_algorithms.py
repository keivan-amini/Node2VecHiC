"""
Tests related to the algorithms module.
"""

from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from test_metadata import generate_metadata
from test_hic import generate_hic
from Node2VecHiC.algorithms import run_node2vec, get_embeddings, run_pca, normalize_features
from Node2VecHiC.metadata import Metadata
from Node2VecHiC.hic import HiC

np.random.seed(123)

# Defining instance of the classes
METADATA_PATH = generate_metadata()
HIC_PATH = generate_hic()
metadata = Metadata(METADATA_PATH)
hic = HiC(metadata, HIC_PATH)


def generate_parameters() -> (str, list):
    """
    Generate Node2Vec parameters

    Returns
    -------
        embeddings_path (str):
            string containing the embeddings path
        parameters (list):
            parameters of the Node2Vec algorithm

    """
    embeddings_path = 'embeddings.csv'
    N_DIMENSIONS = 3
    WALK_LENGTH = 25
    NUM_WALKS = 3
    WEIGHT_KEY = 'weight'
    WORKERS = 1
    P = 1
    Q = 0.5
    parameters = [N_DIMENSIONS, WALK_LENGTH, NUM_WALKS, WEIGHT_KEY, WORKERS, P, Q]
    return embeddings_path, parameters



########## TEST FUNCTIONS ##########

def test_parameters_node2vec():
    """
    GIVEN: a set of parameters and an hic graph
    WHEN: applying the function run_node2vec()
    THEN: the data types input parameters are correct.
    """
    _ , parameters = generate_parameters()
    data_types = [int, int, int, str, int, int, float]
    for parameter, data_type in zip(parameters, data_types):
        assert isinstance(parameter, data_type)


def test_embeddings_node2vec():
    """
    GIVEN: a graph
    WHEN: applying the function run_node2vec()
    THEN: the algorithm returns a number of embeddings
    with the same number of nodes of the starting graph.
    """
    graph = hic.graph
    EMBEDDINGS_PATH, parameters = generate_parameters()
    embeddings, _ = run_node2vec(graph,
                                 parameters,
                                 EMBEDDINGS_PATH)
    assert graph.number_of_nodes() == len(embeddings)


def test_function_run_node2vec():
    """
    GIVEN: a graph
    WHEN: applying the run_node2vec() function
    THEN: the output embeddings from the function are
    the same with respect to the expected embeddings.
    """
    graph = hic.graph
    EMBEDDINGS_PATH, parameters = generate_parameters()
    embeddings, _ = run_node2vec(graph,
                                 parameters,
                                 EMBEDDINGS_PATH)
    expected_data = {0: [0.231323, -0.124257, 0.185762, 0.325982, 0.013314, -0.107135, -0.253756, -0.054519],
                     1: [0.381421, 0.317647, 0.295100, -0.199960, 0.119814, 0.329491, -0.233661, 0.143450],
                     2: [-0.187161, -0.198803, 0.006390, -0.281146, 0.127886, -0.085097, 0.211660, 0.021345]}
    expected_df = pd.DataFrame(expected_data)
    assert np.allclose(embeddings, expected_df)

def test_parameter_get_embeddings():
    """
    GIVEN: a set of parameters for the node2vec algorithm
    WHEN: applying the get_embeddings() function
    THEN: the n_dimensions parameter and the path parameter
    are compatible, i.e., the path containing the learned embeddings
    encodes a dataframe with 'n_dimensions' column.
    """
    EMBEDDINGS_PATH, parameters = generate_parameters()
    n_dimensions = parameters[0]
    embeddings_df, _ = get_embeddings(n_dimensions,
                                      EMBEDDINGS_PATH)
    assert len(embeddings_df.columns) == n_dimensions

@given(n_components = st.integers(min_value = 1,
                                  max_value = len(hic.data_frame.columns)-1))
def test_rows_pca(n_components):
    """
    GIVEN: an input dataframe and a generated number of
    principal components
    WHEN: applying the run_pca() function
    THEN: the number of rows in dataframes before and
    after PCA is preserved.
    """
    input_df = hic.data_frame
    output_df = run_pca(input_df,
                        n_components)
    assert len(input_df.index) == len(output_df.index)

def test_dim_retention_pca():
    """
    GIVEN: an input data frame and a number of components equal
    to the data frame dimension
    WHEN: applying the run_pca() function
    THEN: the output data frame is equal to the input data
    frame, i.e., PCA is not executed.
    """
    input_df = hic.data_frame
    n_components = len(input_df.columns)
    output_df = run_pca(input_df,
                        n_components)
    assert output_df.equals(input_df)

def test_function_run_pca():
    """
    GIVEN: a dataframe example and a number of components
    equals two
    WHEN: applying the run_pca() function
    THEN: the output result equals the expected result, under
    a tolerance of 1e-5.
    """
    data = {0: [0, 1, 2, 9, 10, 11],
            1: [4, 7, 9, 4, 1, 6]}
    input_df = pd.DataFrame(data)
    output_df = run_pca(input_df,
                        n_components = 2)
    expected_data = {0: [-0.526089, -1.205312, -1.606591, 0.865405, 1.853850, 0.618737],
                     1: [1.174626, 0.186182, -0.524318, -0.216868, 0.462356, -1.081978]}
    expected_df = pd.DataFrame(expected_data)
    assert np.allclose(output_df, expected_df)


@given(n_components = st.integers(min_value = 1,
                                  max_value = len(hic.data_frame.columns)-1))
def test_parameter_pca(n_components):
    """
    GIVEN: the parameter n_components
    WHEN: applying the function run_pca()
    THEN: the n_components integer parameter equals the 
    dimensionality of the returned dataframe.
    """
    principal_df = run_pca(hic.data_frame, n_components)
    assert len(principal_df.columns) == n_components

def test_mean_normalize_features():
    """
    GIVEN: a set of features
    WHEN: applying the normalize_features() function
    THEN: for each dimension, the set of features has null
    mean under a tolerance of 1e-05. 
    """
    features = normalize_features(hic.data_frame)
    tolerance = 1e-2  # float values can give problems with tests
    means = np.mean(features, axis = 0)
    assert np.all(np.abs(means) < tolerance)

def test_std_dev_normalize_features():
    """
    GIVEN: a set of features
    WHEN: applying the normalize_features() function
    THEN: for each dimension, has normalized standard deviation,
    under a tolerance of 1e-05. 
    """
    features = normalize_features(hic.data_frame)
    tolerance = 1e-05
    std_devs = np.std(features, axis = 0)
    assert np.all(np.abs(std_devs - 1.0) < tolerance)

def test_function_normalize_features():
    """
    GIVEN: a set of example dataframe
    WHEN: applying the normalize_features() function
    THEN: the output result equals the expected result, under
    a tolerance of 1e-05.
    """
    data = {0: [0, 1, 2],
            1: [4, 7, 9]}
    data_frame = pd.DataFrame(data)
    output_array = normalize_features(data_frame)
    expected_array = np.array([[-1.22474487, -1.29777137],
                               [0.0, 0.16222142],
                               [1.22474487, 1.13554995]])
    assert np.allclose(output_array, expected_array)
