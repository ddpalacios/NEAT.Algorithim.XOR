import unittest
from NEAT import NEAT
from Node import Node


class TestConnectionGenes(unittest.TestCase):

    def setUp(self):
        self.neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        self.genome = self.neat.generate_empty_genome()
        self.initial_connection_length = len(self.genome.connection_genes)
        self.initial_node_length = len(self.genome.nodes)
        self.genome.add_connection_mutation()

    def test_add_connection(self):
        self.assertEqual(len(self.genome.connection_genes), self.initial_connection_length + 1)

    def test_valid_connection(self):
        self.genome.add_node_mutation()

        node1 = self.genome.nodes[0]  # bias
        node2 = self.genome.nodes[1]  # input
        self.assertFalse(self.genome.is_valid_connection(node1, node2)[0])

        node1 = self.genome.nodes[1]  # input
        node2 = self.genome.nodes[1001]  # hidden
        self.assertTrue(self.genome.is_valid_connection(node1, node2)[0])

        node1 = self.genome.nodes[1]  # input
        node2 = self.genome.nodes[3]  # output
        self.assertTrue(self.genome.is_valid_connection(node1, node2)[0])

        # existing connection
        node1 = self.genome.connection_genes[1].in_node
        node2 = self.genome.connection_genes[1].out_node
        self.assertFalse(self.genome.is_valid_connection(node1, node2)[0])

    def test_incoming_and_outgoing_connections(self):
        connection_gene = self.genome.connection_genes[1]
        in_node = connection_gene.in_node
        out_node = connection_gene.out_node
        self.assertEqual(1, len(in_node.outgoing_connections))
        self.assertEqual(1, len(out_node.incoming_connections))

    def test_has_weight(self):
        self.assertTrue(self.genome.connection_genes[1].weight is not None)
        print('Weight', self.genome.connection_genes[1].weight)

    def test_add_node(self):
        hidden_node = self.genome.add_node_mutation()

        self.assertEqual(len(self.genome.nodes), self.initial_node_length + 1)
        self.assertEqual(3, len(self.genome.connection_genes))
        self.assertFalse(self.genome.connection_genes[1].is_enabled)
        self.assertEqual(1, self.genome.connection_genes[2].weight)
        self.assertEqual(self.genome.connection_genes[1].weight, self.genome.connection_genes[3].weight)

        # hidden outgoing and incoming connections length
        self.assertEqual(1, len(hidden_node.incoming_connections))
        self.assertEqual(1, len(hidden_node.outgoing_connections))
        self.assertEqual(2, len(self.genome.nodes[3].incoming_connections))


if __name__ == '__main__':
    unittest.main()
