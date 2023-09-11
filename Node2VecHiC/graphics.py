"""
This script has the aim to construct the Graphics() class,
useful to visualize features emerging after the application
of algorithms on the adjacency matrix.
"""

from random import randint
import matplotlib.pyplot as plt
import pandas as pd

# per la grafica: fai funzione che prende in input una lista di dataframe e realizza tanti plot quanti sono i df!
#TODO to finish docstring and nice code!!!!!!!!!!!!


class Graphics:

    """
    Description
    -----------

    
    Methods
    -------

    Attributes
    ----------
    
    
    """


    def __init__(self, hi_c):

        self.hi_c = hi_c
        self.metadata_df = self.hi_c.metadata.data_frame
        self.attributes = self.hi_c.attributes
        self.name = self.hi_c.name

        self.number_chromosomes = len(self.metadata_df)

        self.color_list = self.get_colors_list()
        self.color_dict = self.get_color_dict()

    def get_colors_list(self) -> list:
        """
        Starting with a list of predefined colors, add colors
        randomly in order to get a list in which each color is
        associated to a couple of chromosomes. 

        Return
        ------
            colors (list):
                list containing colors related to chromosomes.
                TODO to tests len == self.number_chromosomes
        
        """
        predefined_colors = ['tab:blue', 'tab:green', 'tab:red',
                             'tab:orange', 'tab:pink'] # honestly, colors I like!
        random_colors = [f'#{randint(0, 0xFFFFFF):06X}'
                         for _ in range(self.number_chromosomes - len(predefined_colors))]
        colors = predefined_colors + random_colors
        return colors


    def get_color_dict(self) -> dict:
        """
        Get a dictionary that assign each chromosome to a color.

        Return
        ------
            color_dict (dict):
                dictionary containing as chromosome : color
                as key : value.
        
        """
        chromosomes_list = self.metadata_df.chr.tolist()
        color_dict = {chromosomes_list[index]: self.color_list[index]
                      for index in range(len(chromosomes_list))}
        return color_dict


    def get_plot_chromosome(self,
                            principal_df: pd.DataFrame,
                            save: bool = False):
        """
        Show scatter visualization of principal components in which
        each scatter point is colorized with respect to the associated
        chromosome.


        Parameters
        ----------
            principal_df (pd.DataFrame):
                data frame containing principal components, obtained
                after performing a PCA on a certain dataset.
                Maximum number of components accepted is 3. TODO tests
            save (bool):
                boolean variable. If true, save the visualization in
                the plot folder. 
        """

        fig, axes = plt.subplots()
        axes.set_axisbelow(True)
        axes.grid()

        n_components = len(principal_df.columns)

        for index, obs in principal_df.iterrows():
            if index in self.attributes:
                components = obs[:n_components]
                axes.scatter(*components, c = self.color_dict[self.attributes[index]],
                        s = 7, label = str(self.attributes[index]))

        axes_labels = [f"Principal Component {i+1}" for i in range(min(n_components, 3))]

        for i, label in enumerate(axes_labels):
            getattr(axes, ['set_xlabel', 'set_ylabel', 'set_zlabel'][i])(label)

        axes.legend()
        axes.set_title(str(self.name) + ' network.')
        legend_without_duplicate_labels(axes)
        if save:
            fig.savefig('..\\plot\\pca_chromosome_' + str(self.name))
        plt.show()


    def get_plot_blocks(self,
                        principal_block_dfs: list,
                        selected_chromosome: int, #TODO docstrings for selected_chromosomes
                        save: bool = True):
        """
        pisello
        
        """
        
        fig, axes = plt.subplots(nrows = 1,
                                 ncols = len(principal_block_dfs),
                                 figsize = (16, 4))

        string_selected_chromosome = (self.attributes[selected_chromosome]) #TODO c'è chiaramente un problema!!!

        #fig.suptitle(str(self.name))

        n_components = max(len(df.columns) for df in principal_block_dfs)

        for axes_index, ax in enumerate(axes):
            ax.set_axisbelow(True)
            ax.grid()
            df = principal_block_dfs[axes_index]
            for obs_index, obs in df.iterrows():
                if obs_index in self.attributes:
                    components = obs[:n_components]
                    ax.scatter(*components, c = self.color_dict[self.attributes[obs_index]],
                            s = 7, label = str(self.attributes[obs_index]))
            ax.legend()
            legend_without_duplicate_labels(ax)
            #ax.set_title(f'Subplot {axes_index + 1}')  # Set titles for each subplot


        

        plt.tight_layout()
        if save:
            fig.savefig('..\\plot\\blocks\\' + str(self.name) + '_' + string_selected_chromosome)
        plt.show()
 

























    def get_plot_chromosome_comparison(self, df1, df2): #non voglio self. voglio che self.attributes sia variabile globale. non mi piace
            fig, axes = plt.subplots() 
            axes.set_axisbelow(True)
            axes.grid()
            for index, x in df1.iterrows():
                axes.scatter(x[0], x[1], c = self.color_dict[self.attributes[index]], marker = 'x', s = 30, label = str(self.attributes[index]))  #tumor
            for index, x in df2.iterrows():
                axes.scatter(x[0], x[1], c = self.color_dict[self.attributes[index]], marker = 'v', s = 30, label = str(self.attributes[index])) #healthy
            axes.set_xlabel("First principal component")
            axes.set_ylabel("Second principal component")
            #aggiungere un ulteriore legenda con la distinzione tra tumore e cellula sana
            axes.set_title('Chromosome comparison for healthy and tumor cell with PCA and Node2Vec')
            legend_without_duplicate_labels(axes)
            fig.savefig('..\plot\pca_chromosome_comparison' + '-' + str(self.D) + 'dim.png') #cambiare nome
            plt.show() #avere il grafico a destra e avere una sola legenda










    
"""
def get_comparison_blocks(principal_components):
    fig, axes = plt.subplots()
    axes.set_axisbelow(True)
    axes.grid()
    x1, y1 = principal_components[:,0], principal_components[:,1]
    axes.scatter(x1, y1, color = "tab:red", s= 4, label = 'Cancer Hi-C')
"""




# ci potrebbe stare fare un plottino per tante dimensioni del problema node2vec

# di questa funzione, per adesso non me ne frega un cazzo proprio. e non funziona nemmeno perchè selfD non è qui
def get_plot(features1, features2): #non mi piace il fatto che features1 debba essere per forza cancro
    fig, axes = plt.subplots() 
    axes.set_axisbelow(True)
    axes.grid()
    x1, y1 = features1[:,0], features1[:,1]
    x2, y2 = features2[:,0], features2[:,1]
    axes.scatter(x1, y1, color = "tab:red", s= 4, label = 'Cancer Hi-C') # da ottimizzare
    axes.scatter(x2, y2, color = "tab:green", s = 4, label = 'Healthy Hi-C')
    axes.set_title('PCA on node2vec algorithm')
    axes.set_xlabel("First principal component")
    axes.set_ylabel("Second principal component")
    axes.legend()
    #fig.savefig('..\plot\pca' + '-' + str(self.D) + 'dim.png') #cambiare nome
    plt.show()

def legend_without_duplicate_labels(axes: object):
    """
    Create a legend for a Matplotlib axes without duplicate labels.

    Parameter
    ---------
        axes (object):
            The Matplotlib axes object for which to create the legend.

    """
    handles, labels = axes.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    axes.legend(*zip(*unique))






""" Generate different images: 1,2,3,4 with blocks_df
# Iterate through the specified timesteps and visualize the images
for index, i in enumerate([10, 50, 100, 185]):
    # Generate noisy image and noise using forward_noising function
    ## ToDo
    noisy_im, noise = forward_noising(jax.random.PRNGKey(42),
                                      sample_mnist,
                                      i,
                                      sqrt_alpha_bar,
                                      one_minus_sqrt_alpha_bar)

    # Plot the noisy image
    plt.subplot(1, 4, index + 1)
    plt.imshow(jnp.squeeze(jnp.squeeze(noisy_im, -1), 0), cmap='gray')
    plt.title(f"Timestamp {i}")
    plt.axis('off')

# Show the figure with the visualized images
plt.show()


"""
