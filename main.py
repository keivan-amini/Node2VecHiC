from hic import HiC

cancer_path = 'cancer_hic.csv'
healthy_path = 'healthy_hic.csv'

cancer, healthy = HiC(cancer_path),  HiC(healthy_path)