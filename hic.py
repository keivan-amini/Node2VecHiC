import newtworkx as nx
import pandas as pd

class HiC:

    def __init__(self, path):
        self.path = path

    def get_df(self):
        self.df = pd.read_csv(self.path)

    def get_network(self):
        Graphtype = nx.Graph()
        self.G = nx.from_pandas_edgelist(self.df, edge_attr='weight',
                                         create_using = Graphtype)



