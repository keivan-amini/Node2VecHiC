import numpy as np
from hic import HiC, Algorithms, get_plot

cancer_path = '..\data\cancer_hic.csv' 
healthy_path = '..\data\healthy_hic.csv'
metadata_path = '..\data\metadata_hic.xlsx' 


np.random.seed(1)
cancer_hic = HiC(cancer_path, metadata_path)
healthy_hic = HiC(healthy_path, metadata_path)

algorithms_cancer = Algorithms(cancer_hic)
algorithms_healthy = Algorithms(healthy_hic)

algorithms_cancer.node2vec()
algorithms_healthy.node2vec()

algorithms_cancer.pca()
algorithms_healthy.pca()


x_cancer = algorithms_cancer.new_x
x_healthy = algorithms_healthy.new_x

get_plot(x_cancer, x_healthy)


