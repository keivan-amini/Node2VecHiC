"""
This module contains functions related to the applications of
algorithms to dataframe extracted from adjacency matrices.
Specifically, the implemeted functions are designed to work on
attributes of the Hi-C() class.
At the moment, node2vec and pca algorithms have been been implemented
relying on already developed python libraries.
"""

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#----------------------------------------------------------

# Node2Vec

def run_node2vec(graph: nx.Graph,
                 parameters: list,
                 embeddings_path: str) -> (pd.DataFrame, list):
    """
    Perform the node2vec algorithm with the chosen parameters
    on the input graph, fit the learned model and save the
    embeddings on the desired path. Return the dataframe
    containing the node embeddings.

    Parameters
    ----------
        graph (nx.Graph):
            graph structure on which the node2vec algorithm
            is applied.
        parameters (list):
            list containing the following Node2Vec parameters
            [n_dimensions, walk_length, num_walks, weight_key,
            workers, p, q]

            Elements
            --------
                n_dimensions (int):
                    desired dimensions of the embedding space.
                walk length (int):       
                    Length of random walks performed by Node2Vec.
                num_walks (int):
                    Number of random walks to be generated per node.
                weight_key (str):
                    Edge attribute key used as the weight when sampling
                    neighbors.
                workers (int):
                    Number of CPU cores to use for training the model.
                p (float):
                    Node2Vec parameter controlling the likelihood of
                    revisiting previous nodes in random walks.
                q (float):
                    Node2Vec parameter controlling the likelihood of
                    exploring new nodes in random walks.

                For further informations please look at the repository
                exploited for the actual implementation of the algorithm
                'https://github.com/eliorc/node2vec'.        
            
        embeddings_path (string):
            string representing the desired path in which
            the output embeddings is saved.


    Return
    ------
        data_frame (pd.DataFrame):
            dataframe containing the saved node2vec generated
            embeddings.
        indexes (list)
            indexes nodes associated with the learned embeddings.
            
        
    """
    n_dimensions, walk_length, num_walks, weight_key, workers, \
    p, q = parameters
    node2vec = Node2Vec(graph = graph,
                        dimensions = n_dimensions,
                        walk_length = walk_length,
                        num_walks = num_walks,
                        weight_key = weight_key,
                        workers = workers,
                        p = p,
                        q = q)
    model = fit_model(node2vec)
    model.wv.save_word2vec_format(embeddings_path)
    data_frame, indexes = get_embeddings(n_dimensions, embeddings_path)
    return data_frame, indexes

def fit_model(node2vec: object,
              window: int = 10,
              min_count: int = 1,
              batch_words: int = 4) -> object:
    """
    Fit the Node2Vec model with specified parameters.
    Return the model.

    Parameters
    ----------
        node2vec (object):
            Instance of the Node2Vec() class.
        window (int), optional:
            Maximum distance between the current and predicted
            nodes in the skip-gram model. Default is 10.
        min_count (int), optional:
            Ignores all words (nodes) with a total frequency
            lower than this. Default is 1.
        batch_words (int, optional):
            Number of words (nodes) in each batch of words used
            for training. Default is 4.

    Return
    ------
        model (object):
            The fitted Node2Vec model.
    """
    model = node2vec.fit(window = window,
                         min_count = min_count,
                         batch_words = batch_words)
    return model


def get_embeddings(n_dimensions: int,
                   path: str) -> (pd.DataFrame, list):
    """
    Get the dataframe containing the node embeddings and
    sort it by the indexes. Return both the dataframe
    and the indexes nodes associated with the learned
    embeddings.

    Parameter
    ---------
        n_dimensions (int):
            desired dimensions of the embedding space.
        path (string):
            string representing the desired path in which
            the output embeddings are saved.
    Return
    ------
        embeddings (pd.DataFrame)
            ordered dataframe containing the learned embeddings.
        indexes (list)
            indexes nodes associated with the learned embeddings.
    """
    dimensions = np.arange(n_dimensions)
    embeddings = pd.read_csv(path, header = 0,
                             names = dimensions, # specify names, otherwise columns are messed up
                             sep = ' ')
    embeddings.sort_index(inplace = True)
    indexes = embeddings.index.to_list()
    return embeddings, indexes



#----------------------------------------------------------

# PCA

def run_pca(data_frame: pd.DataFrame,
            n_components: int) -> np.ndarray:
    """
    Normalize features on a input dataframe and run PCA algorithm
    with the desired number of prinicpal components.

    Parameters
    ----------
        data_frame (pd.DataFrame):
            input dataframe containing multi-dimensional features.
        n_components (int):
            number of principal components to retain after dimensionality reduction.

    Return
    ------
        components (np.ndarray):
            array of shape (n_samples, n_features_new) containing
            the extracted principal components.

    """
    dimensions = np.arange(n_components)
    indexes = data_frame.index.to_list()
    features = normalize_features(data_frame)
    pca = PCA(n_components)
    principal_components = pca.fit_transform(features)
    principal_df = pd.DataFrame(data = principal_components,
                                columns = dimensions,
                                index = indexes) # indexes is necessary if we remove empy axes!
    #components = principal_df.loc[:, dimensions].values
    return principal_df #TODO change docstrings since we now just return principal_df


def normalize_features(data_frame: pd.DataFrame) -> np.ndarray:
    """
    Normalize features contained in the input dataframe.

    Parameters
    ----------
        data_frame (pd.DataFrame):
            data_frame from which we want to normalize
            features.
    Return
    ------
        normalized_features (np.ndarray):
            array of shape of shape (n_samples, n_features_new)
            containing the normalized features.
    """

    dimensions = np.arange(len(data_frame.columns))
    features = data_frame.loc[:, dimensions].values
    normalized_features = StandardScaler().fit_transform(features)
    return normalized_features
