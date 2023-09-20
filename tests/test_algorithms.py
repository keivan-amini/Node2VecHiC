"""
Tests related to the algorithms module.
"""

from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from test_metadata import generate_metadata
from test_hic import generate_hic
from Node2VecHiC.algorithms import run_node2vec, fit_model, get_embeddings, run_pca, normalize_features
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


def test_parameters_node2vec():
    """
    GIVEN: a set of parameters and an hic graph
    WHEN: applying the function run_node2vec()
    THEN: the data types input parameters are correct.
    """
    EMBEDDINGS_PATH, parameters = generate_parameters()
    data_types = [int, int, int, str, int, int, float]
    for parameter, data_type in zip(parameters, data_types):
        print(parameter, type(parameter))
        assert isinstance(parameter, data_type)


def test_parameter_get_embeddings():
    """
    GIVEN: a set of parameters for the node2vec algorithm
    WHEN: applying the get_embeddings() function
    THEN: the n_dimensions parameter and the path parameter
    are compatible, i.e., the path containing the learned embeddings
    encodes a dataframe with 'n_dimensions' column
    """
    EMBEDDINGS_PATH, parameters = generate_parameters()
    n_dimensions = parameters[0]
    embeddings_df, _ = get_embeddings(n_dimensions,
                                      EMBEDDINGS_PATH)
    assert len(embeddings_df.columns) == n_dimensions


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

def test_normalize_features():
    """
    GIVEN: a set of features
    WHEN: applying the normalize_features() function
    THEN: for each dimension, the set of features has mean
    approximately equal to 0 and equivalently standard deviation
    equal to 1. 
    """
    features = normalize_features(hic.data_frame)
    tolerance = 1e-2  # float values can give problems with tests
    means = np.mean(features, axis = 0)
    std_devs = np.std(features, axis = 0)
    assert np.all(np.abs(means) < tolerance)
    assert np.all(np.abs(std_devs - 1.0) < tolerance)
    