"""
Main script controlling the execution of the different modules.
Simulations useful to reproduce the plots contained in Complex Network project (see .pdf folder)
"""
import numpy as np

from Node2VecHiC.metadata import Metadata
from Node2VecHiC.hic import HiC
from Node2VecHiC.algorithms import run_node2vec, run_pca
from Node2VecHiC.graphics import Graphics

np.random.seed(1)

METADATA_PATH = '..\\data\\metadata_hic.xlsx'
CANCER_PATH = '..\\data\\cancer_hic.csv'


metadata = Metadata(METADATA_PATH)
cancer_hic = HiC(metadata, CANCER_PATH)
cancer_graph = cancer_hic.graph

"""
################################################## WALK LENGTH TEST: 10,100,1000,1500
# Node2Vec parameters
N_DIMENSIONS = 10 
NUM_WALKS = 10 
WEIGHT_KEY = 'weight'
WORKERS = 1 
P = 1 
Q = 0.5 

WALK_LENGTH = [10, 100,1000,1500]
number_plots = 4
#PCA
N_COMPONENTS = 2

cancer_graphics = Graphics(cancer_hic)

for index in range(number_plots):
    EMBEDDINGS_PATH = '..\\model\\walk_length_' + str(index) + '.csv'
    wl = WALK_LENGTH[index]
    print("walk_length = " + str(wl))
    parameters = [N_DIMENSIONS, wl, NUM_WALKS, WEIGHT_KEY, WORKERS, P, Q]


    embeddings, indexes = run_node2vec(cancer_graph,
                                   parameters,
                                   EMBEDDINGS_PATH)
    principal_df = run_pca(data_frame=embeddings,
                           n_components=N_COMPONENTS)
    cancer_graphics.get_plot_chromosome(principal_df,
                                        title = "Walk length = " + str(wl),
                                        save_path= '..\\plot\\walk_length\\' + str(wl)
"""
"""
################################################### NUMBER WALK TEST: 10,100,1000,1500
# Node2Vec parameters
N_DIMENSIONS = 10
WALK_LENGTH = 10 
WEIGHT_KEY = 'weight'
WORKERS = 1
P = 1 
Q = 0.5 


NUM_WALKS = [10, 100, 1000, 1500]
number_plots = 4
#PCA
N_COMPONENTS = 2

cancer_graphics = Graphics(cancer_hic)

for index in range(number_plots):
    EMBEDDINGS_PATH = '..\\model\\num_walks_' + str(index) + '.csv'
    nw = NUM_WALKS[index]
    print("num_walks = " + str(nw))
    parameters = [N_DIMENSIONS, WALK_LENGTH, nw, WEIGHT_KEY, WORKERS, P, Q]


    embeddings, indexes = run_node2vec(cancer_graph,
                                   parameters,
                                   EMBEDDINGS_PATH)
    principal_df = run_pca(data_frame=embeddings,
                           n_components=N_COMPONENTS)
    cancer_graphics.get_plot_chromosome(principal_df,
                                        title = "Number walks = " + str(nw),
                                        save_path= '..\\plot\\num_walks\\' + str(nw))
"""
"""
###################################################  p and q values for a BFS-like exploration strategy


    p = 2.0, q = 0.5
    p = 1.5, q = 0.75
    p = 2.0, q = 0.8
    p = 1.8, q = 0.9
    p = 1.6, q = 0.6
"""

"""
# Node2Vec parameters
N_DIMENSIONS = 10
WALK_LENGTH = 10 
WEIGHT_KEY = 'weight'
WORKERS = 1
P = [2,1.5,2.0,1.8,1.6] 
Q = [0.5, 0.75, 0.8, 0.9, 0.6]
NUM_WALKS = 10


number_plots = 5
#PCA
N_COMPONENTS = 2

cancer_graphics = Graphics(cancer_hic)

for index in range(number_plots):
    EMBEDDINGS_PATH = '..\\model\\bfs_' + str(index) + '.csv'
    pi = P[index]
    qi = Q[index]
    print("P = " + str(pi) + ". Q = " + str(qi))
    parameters = [N_DIMENSIONS, WALK_LENGTH, NUM_WALKS, WEIGHT_KEY, WORKERS, pi, qi]

    embeddings, indexes = run_node2vec(cancer_graph,
                                   parameters,
                                   EMBEDDINGS_PATH)
    principal_df = run_pca(data_frame=embeddings,
                           n_components=N_COMPONENTS)
    cancer_graphics.get_plot_chromosome(principal_df,
                                        title = "BFS strategy. P = " + str(pi) + ", Q = " + str(qi),
                                        save_path= '..\\plot\\bfs\\' + str(pi) + '_' + str(qi)+'.png')


"""

###################################################  p and q values for a depth-first sampling DFS strategy using Node2Vec:
"""
    p = 0.5, q = 2.0
    p = 0.75, q = 1.5
    p = 0.3, q = 1.8
    p = 0.8, q = 1.7
    p = 0.6, q = 2.2

"""
"""
# Node2Vec parameters
N_DIMENSIONS = 10
WALK_LENGTH = 10 
WEIGHT_KEY = 'weight'
WORKERS = 1
P = [0.5, 0.75, 0.3, 0.8, 0.6] 
Q = [2.0, 1.5, 1.8, 1.7, 2.2]
NUM_WALKS = 10


number_plots = 5
#PCA
N_COMPONENTS = 2

cancer_graphics = Graphics(cancer_hic)

for index in range(number_plots):
    EMBEDDINGS_PATH = '..\\model\\dfs_' + str(index) + '.csv'
    pi = P[index]
    qi = Q[index]
    print("P = " + str(pi) + ". Q = " + str(qi))
    parameters = [N_DIMENSIONS, WALK_LENGTH, NUM_WALKS, WEIGHT_KEY, WORKERS, pi, qi]

    embeddings, indexes = run_node2vec(cancer_graph,
                                   parameters,
                                   EMBEDDINGS_PATH)
    principal_df = run_pca(data_frame=embeddings,
                           n_components=N_COMPONENTS)
    cancer_graphics.get_plot_chromosome(principal_df,
                                        title = "DFS strategy. P = " + str(pi) + ", Q = " + str(qi),
                                        save_path= '..\\plot\\dfs\\' + str(pi) + '_' + str(qi)+'.png')

"""

######################################## DIMENSIONS = [10,50,100,1000]
# Node2Vec parameters
N_DIMENSIONS = [10,50,100,1000]
WALK_LENGTH = 10 
WEIGHT_KEY = 'weight'
WORKERS = 1
P = 1
Q = 0.5
NUM_WALKS = 10


number_plots = 4
#PCA
N_COMPONENTS = 2

cancer_graphics = Graphics(cancer_hic)

for index in range(number_plots):
    EMBEDDINGS_PATH = '..\\model\\dim_' + str(index) + '.csv'
    dim = N_DIMENSIONS[index]
    print("dim = " + str(dim))
    parameters = [dim, WALK_LENGTH, NUM_WALKS, WEIGHT_KEY, WORKERS, P, Q]

    embeddings, indexes = run_node2vec(cancer_graph,
                                   parameters,
                                   EMBEDDINGS_PATH)
    principal_df = run_pca(data_frame=embeddings,
                           n_components=N_COMPONENTS)
    cancer_graphics.get_plot_chromosome(principal_df,
                                        title = "Number dimensions = " + str(dim) + " before PCA",
                                        save_path= '..\\plot\\dim\\' + str(dim))
    