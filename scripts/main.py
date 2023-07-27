import numpy as np
from hic import HiC, Algorithms
from hic import get_plot


cancer_path = '..\data\cancer_hic.csv' 
healthy_path = '..\data\healthy_hic.csv'
metadata_path = '..\data\metadata_hic.xlsx' 


np.random.seed(1)
cancer_hic = HiC(cancer_path, metadata_path)
healthy_hic = HiC(healthy_path, metadata_path)

attributes_cancer = cancer_hic.attributes



algorithms_cancer = Algorithms(cancer_hic)
algorithms_healthy = Algorithms(healthy_hic)

#algorithms_cancer.node2vec()
#algorithms_healthy.node2vec()

algorithms_cancer.get_df()
algorithms_healthy.get_df()

algorithms_cancer.normalize_features()
algorithms_healthy.normalize_features()

algorithms_cancer.pca()
algorithms_healthy.pca()


x_cancer = algorithms_cancer.new_x
x_healthy = algorithms_healthy.new_x

prinicpal_df1 = algorithms_cancer.principal_df
principal_df2 = algorithms_healthy.principal_df

get_plot(x_cancer, x_healthy)

algorithms_cancer.get_plot_chromosome(type= 'cancer')

algorithms_healthy.get_plot_chromosome(type= 'healthy')

algorithms_cancer.get_plot_chromosome_comparison(prinicpal_df1, principal_df2)

