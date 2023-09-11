"""
Main script controlling the execution of the different modules.
"""
import numpy as np

from metadata import Metadata
from hic import HiC
from algorithms import run_node2vec, run_pca, get_embeddings
from graphics import Graphics

np.random.seed(1)

METADATA_PATH = '..\\data\\metadata_hic.xlsx'
CANCER_PATH = '..\\data\\cancer_hic.csv'
HEALTHY_PATH = '..\\data\\healthy_hic.csv'

EMBEDDINGS_PATH = '..\\model\\blocks\\cancer_20-1.csv'
EMBEDDINGS_PATH1 = '..\\model\\blocks\\cancer_20-6.csv' #IN REALTà 6-10 MA FACCIAMO UNA PROVA E VEDIAMO SE IL MODELLO è UGUALE!!!!
EMBEDDINGS_PATH2 = '..\\model\\blocks\\cancer_20-X.csv' # da cambiare. attento ai nomi
EMBEDDINGS_PATH3 = '..\\model\\blocks\\cancer_20-10.csv'


metadata = Metadata(METADATA_PATH)
cancer_hic = HiC(metadata, CANCER_PATH)

#for df in block_dfs:
#    print(df)


# Node2Vec parameters
N_DIMENSIONS = 10 # 10 #attenzione, parametro da cambiare ogni volta che non si genera node2vec ma che si prende direttamente il df dal folder model!!
WALK_LENGTH = 300 #300
NUM_WALKS = 10 #10
WEIGHT_KEY = 'weight'
WORKERS = 1
P = 1
Q = 0.5

parameters = [N_DIMENSIONS, WALK_LENGTH, NUM_WALKS, WEIGHT_KEY, WORKERS, P, Q,]

"""
cancer_graph = cancer_hic.graph

embeddings, indexes = run_node2vec(cancer_graph,
                                   parameters,
                                   EMBEDDINGS_PATH)
"""


#Blocks


graph_block_lists = cancer_hic.get_block_graph()
embeddings_path = [EMBEDDINGS_PATH, EMBEDDINGS_PATH1, EMBEDDINGS_PATH2, EMBEDDINGS_PATH3]
embeddings = []

for idx, graph in enumerate(graph_block_lists):
    embeddings_found, _ = run_node2vec(graph,
                                       parameters,
                                       embeddings_path[idx])
    embeddings.append(embeddings_found)

"""




#OR

embeddings, indexes = get_embeddings(N_DIMENSIONS, EMBEDDINGS_PATH)
embeddings1, indexes1 = get_embeddings(N_DIMENSIONS, EMBEDDINGS_PATH1)
embeddings2, indexes2 = get_embeddings(N_DIMENSIONS, EMBEDDINGS_PATH2)
embeddings3, indexes3 = get_embeddings(N_DIMENSIONS, EMBEDDINGS_PATH3)

 # forse non è necessario far tornare indexes: da vedere se lo è in grafica!!
"""


#PCA
N_COMPONENTS = 2

block = []
selected_chromosome = cancer_hic.selected_chromosome
for model in embeddings:
    principal_df = run_pca(data_frame=model,
                           n_components=N_COMPONENTS)
    block.append(principal_df)





# Graphics

cancer_graphics = Graphics(cancer_hic)
#cancer_graphics.get_plot_chromosome(principal_df) #non mi fa impazzire. components dovrebbe essere dento cancer_graphics.
cancer_graphics.get_plot_blocks(block, selected_chromosome)

# Blocks (notebooks?)

"""
graphs_list = cancer_hic.get_block_graph()
embeddings_path = []
for index in range(len(graphs_list)):
    embeddings_path.append('..\\model\\blocks\\cancer_hic_embedding-block-' + str(index) + '.csv')

principal_blocks_df = []
for graph in graphs_list:
    embeddings, _ = run_node2vec(graph,
                                 parameters,
                                 embeddings_path) # to understand how to put the embeddings path
    principal_df = run_pca(embeddings,
                           N_COMPONENTS)
    principal_blocks_df.append(principal_df)
    
cancer_graphics.get_plot_blocks(principal_blocks_df)
"""
    


    

