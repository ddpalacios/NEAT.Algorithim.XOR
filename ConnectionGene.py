import numpy as np


class ConnectionGene:
    def __init__(self, connection_gene_id, in_node, out_node, weight=np.random.randn()):
        self.in_node = in_node
        self.out_node = out_node
        self.is_enabled = True
        self.weight = weight
        self.innovation_number = connection_gene_id
