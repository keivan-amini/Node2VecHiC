import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

#openpyxl needed

class HiC:

    def __init__(self, path_csv, path_metadata):
        self.path = path_csv
        self.path_metadata = path_metadata
        self.name = self.path.replace('..\data\\', '')
        self.name = self.name.replace('.csv', '') #to optimize: do not like the second repetition fo the function


        self.get_df()
        #can add a function to normalize the graph, if it is not already normalized.
        self.get_metadata_df()
        self.remove_empty_axis()
        self.get_graph()

    def get_df(self):
        self.df = pd.read_csv(self.path, header = None)

    def get_metadata_df(self): #slow. to improve!
        self.metadata_df = pd.DataFrame(pd.read_excel(self.path_metadata))

    def remove_empty_axis(self):
        self.df = self.df.loc[:, (self.df != 0).any(axis=0)]
        self.df = self.df.loc[(self.df!=0).any(axis=1)]

    def get_graph(self):
        self.G = nx.from_pandas_adjacency(self.df)
        list = []
        for i in range(len(self.metadata_df.chr)):
            list += ([self.metadata_df.chr[i]] * (self.metadata_df.end[i] - self.metadata_df.start[i]+1))
        dict = {index: value for index, value in enumerate(list)}
        nx.set_node_attributes(self.G, dict, 'chromosome')
        self.attributes = nx.get_node_attributes(self.G, 'chromosome')
        
    def draw_network(self):
        nx.draw(self.G, arrows = True)
        plt.show()



class Algorithms:

    def __init__(self, hi_c):

        self.hi_c = hi_c
        self.G = hi_c.G
        self.name = hi_c.name
        self.attributes = hi_c.attributes


        # Node2Vec parameters
        self.D = 10
        self.P = 1
        self.Q = 0.5
        self.WL = 300
        self.embedding_path = '..\model\-' + str(self.name) + '_embedding.csv'
        self.df = None

        # PCA parameters
        self.old_dimensions = [dimension for dimension in range(self.D)]
        self.number_components = 2
        self.new_dimensions = [dimension for dimension in range(self.number_components)]


    def node2vec(self):
       self.node2vec = Node2Vec(self.G, dimensions = self.D, walk_length = self.WL,
                                num_walks = 10, weight_key='weight', workers = 1,
                                p=self.P, q= self.Q)
       self.model = self.node2vec.fit(window = 10, min_count = 1, batch_words = 4)
       self.model.wv.most_similar('2')  # Output node names are always strings
       self.model.wv.save_word2vec_format(self.embedding_path)
       self.get_df()
       self.normalize_features()

    def get_df(self):
        self.df = pd.read_csv(self.embedding_path, header = 0,
                              names = self.old_dimensions, sep= ' ')
        
    def normalize_features(self):
        self.x = self.df.loc[:, self.old_dimensions].values
        self.x = StandardScaler().fit_transform(self.x)

    def pca(self):
        self.pca = PCA(n_components = self.number_components)
        principalComponents = self.pca.fit_transform(self.x)
        self.principal_df = pd.DataFrame(data = principalComponents,
                                         columns = self.new_dimensions)
        self.new_x = self.principal_df.loc[:,self.new_dimensions].values



def get_plot(features1, features2): #non mi piace il fatto che features1 debba essere per forza cancro
    plt.rc('axes', axisbelow=True)
    plt.grid()
    x1, y1 = features1[:,0], features1[:,1]
    x2, y2 = features2[:,0], features2[:,1]
    plt.scatter(x1, y1, color = "tab:red", s= 4, label = 'Cancer Hi-C') # da ottimizzare
    plt.scatter(x2, y2, color = "tab:green", s = 4, label = 'Healthy Hi-C')
    plt.title('PCA on node2vec algorithm')
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.legend()
    plt.savefig('..\plot\pca.png') #cambiare nome
    plt.show()

def get_plot_chromosome(features, attribute, type):

    fig, ax = plt.subplots() 
    ax.set_axisbelow(True)
    ax.grid()

    chromosomes = list(set((attribute.values())))
    color_list = ['tab:blue','tab:green', 'tab:red', 'tab:orange', 'tab:pink']
    color_dict = {chr: color for chr, color in zip(chromosomes, color_list)}
    color_dict = dict(sorted(color_dict.items()))
    attribute = dict(sorted(attribute.items())) #order by number items

    x1, y1 = features[:,0], features[:,1]
    for index, value in attribute.items():
        if index < len(attribute.items()):
            ax.scatter(x1[index], y1[index], c = color_dict[value], s = 7, label = str(value))
    ax.set_xlabel("First principal component")
    ax.set_ylabel("Second principal component")
    ax.legend()
    ax.set_title(str(type))
    legend_without_duplicate_labels(ax)

    fig.savefig('..\plot\pca_chromosome_' + str(type) + '.png') #cambiare nome
    plt.show()
    
#avere il grafico a destra e avere una sola legenda


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

#mo: grafico con con entrambi cancri e sane, ma sane cerchio e cancri quadratino dolce

def get_plot_chromosome_comparison(features1, features2, attribute):

    fig, ax = plt.subplots() 
    ax.set_axisbelow(True)
    ax.grid()

    x1, y1 = features1[:,0], features1[:,1]
    x2, y2 = features2[:,0], features2[:,1]

    chromosomes = list(set((attribute.values())))
    color_list = ['tab:blue','tab:green', 'tab:red', 'tab:orange', 'tab:pink']
    color_dict = {chr: color for chr, color in zip(chromosomes, color_list)}
    color_dict = dict(sorted(color_dict.items()))
    attribute = dict(sorted(attribute.items())) #order by number items
    for index, value in attribute.items():
        if index < len(attribute.items()):
            ax.scatter(x1[index], y1[index], marker = 'x', c = color_dict[value], s = 7, label = str(value)) #for the tumor
            ax.scatter(x2[index], y2[index], marker = 'v', c = color_dict[value], s = 7) #for the healthy
    ax.set_xlabel("First principal component")
    ax.set_ylabel("Second principal component")
    ax.legend() #aggiungere un ulteriore legenda con la distinzione tra tumore e cellula sana
    ax.set_title('Chromosome comparison for healthy and tumor cell with PCA and Node2Vec')
    legend_without_duplicate_labels(ax)
    fig.savefig('..\plot\pca_chromosome_comparison.png') #cambiare nome
    plt.show()