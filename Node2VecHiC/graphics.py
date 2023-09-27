"""
This script has the aim to construct the Graphics() class,
useful to visualize features emerging after the application
of algorithms on the adjacency matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(123)


class Graphics:

    """
    Description
    -----------
    This class is useful to visualize plots, in order
    to qualitatively check the emergence of clustering
    on principal components data frame obtained after pca
    and node2vec algorithms.

    
    Methods
    -------
        get_colors_list()
            get a list that contains a color for each chromosome
            in the metadata file. First elements have been manually
            chosen, the rest are randomly chosen.
        get_color_dict()
            get a dictionary that maps each chromosome to a color.
        get_plot_chromosome(principal_df, title, save_path)
            given an input a dataframe that must be thought as a
            dataframe containing the principal components, visualize
            a scatter plot. The parameter save determines the eventual
            saving of the picture.
        get_plot_blocks(principal_block_dfs, selected_chromosome, save)
            given an input principal block dataframe and a selected
            chromosome, visualize the scatters plots related to a list of
            graphs.

    Attributes
    ----------
        hi_c (object):
            instance of the class HiC().
        metadata_df (pd.DataFrame):
            dataframe containing the metadata.
        attributes (dict):
            dictionary containing the index nodes related to
            each chromosome.
        name (str):
            name of the instance Hi-C() class.
        number_chromosomes (int):
            number of the involved chromosome in the
            metadata file.
        color_list (list):
            list containing colors, useful for the visualization
            purpose.
        color_dict (dict):
            dict assigning each chromosome to a certain color.

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
        
        """
        predefined_colors = ['tab:blue', 'tab:green', 'tab:red',
                             'tab:orange', 'tab:pink'] # honestly, colors I like!
        random_colors = [f'#{np.random.randint(0, 0xFFFFFF):06X}'
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
                            title: str,
                            save_path: str = None):
        """
        Show scatter visualization of principal components in which
        each scatter point is colorized with respect to the associated
        chromosome.

        Parameters
        ----------
            principal_df (pd.DataFrame):
                data frame containing principal components, obtained
                after performing a PCA on a certain dataset.
                Maximum number of components accepted is 3.
            title (str):
                string representing the title of the plot.
            save_path (str):
                string representing the path in which the plot
                will be saved. Default is None.
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
        axes.set_title(str(title))
        legend_without_duplicate_labels(axes)
        if save_path is not None:
            fig.savefig(save_path)
        plt.show()


    def get_plot_blocks(self,
                        principal_block_dfs: list,
                        selected_chromosome: int,
                        save: bool = False):
        """
        Show block scatter visualization of principal components,
        in which each subplot corresponds to a certain couple of
        chromosomes. At the moment, this function just works in 2-D
        visualization, i.e. when the selected number of principal
        components for the dimensions reduction is 2.

        Parameters
        ----------
            principal_blocks_dfs (list):
                list containing the different blocks dataframes
                after PCA has been performed.
            selected_chromosome (int):
                label of the chromosome protagonist of the list
                block.
            save (bool):
                boolean variable useful to decide if one would
                like to save the picture or not.
                Default is False.

        
        """
        fig, axes = plt.subplots(nrows = 1,
                                 ncols = len(principal_block_dfs),
                                 figsize = (16, 4))

        name_selected_chromosome = self.attributes[selected_chromosome]
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
        plt.tight_layout()
        if save:
            fig.savefig('..\\plot\\blocks\\' + str(self.name) + '_' + name_selected_chromosome)
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
