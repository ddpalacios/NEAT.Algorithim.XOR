import numpy as np
import random
from ConnectionGene import ConnectionGene
from InnovationDatabase import InnovationDatabase
from Node import Node
import math
import copy


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class Genome:
    def __init__(self, genome_id, neat):
        self.neat = neat
        self.species = None
        self.ID = genome_id
        self.fitness = 0.0
        self.shared_fitness = 0.0
        self.nodes = {}
        self.connection_genes = {}

    def print_connection_genes(self):
        print(f"Genome ID: {self.ID}")
        print(f"Fitness: {self.fitness}")
        print(f"Shared Fitness: {self.shared_fitness}")
        print("Connection Genes:")
        for innovation_number, connection_gene in self.connection_genes.items():
            print(f"  Innovation Number: {innovation_number}")
            print(f"    In Node: {connection_gene.in_node.ID}")
            print(f"    Out Node: {connection_gene.out_node.ID}")
            print(f"    Weight: {connection_gene.weight}")
            print(f"    Enabled: {connection_gene.is_enabled}")
            print("    ----------------")
        print("========================================")

    def get_enabled_connections(self):
        enabled_connections = {}
        for connection in self.connection_genes.values():
            if connection.is_enabled:
                enabled_connections[connection.innovation_number] = connection

        return enabled_connections

    def get_input_nodes(self):
        return [node for node in self.nodes.values() if node.type == 'input']

    def get_hidden_nodes(self):
        return [node for node in self.nodes.values() if node.type == 'hidden']

    def get_output_nodes(self):
        return [node for node in self.nodes.values() if node.type == 'output']

    def is_valid_connection(self, node1, node2):
        if node1.rank == node2.rank or node1 == node2:
            return False, None
        if node1.rank > node2.rank:
            node1, node2 = node2, node1

        existing_connection_nodes = {}
        for connection in self.connection_genes.values():
            pair = (connection.in_node, connection.out_node)
            existing_connection_nodes[pair] = connection
        if (node1, node2) in existing_connection_nodes:
            return False, None
        return True, (node1, node2)

    def activate(self, inputs):
        # Set bias node value to 1
        self.nodes[0].value = 1

        # Assign input values to input nodes
        for i, input_value in enumerate(inputs):
            self.nodes[i + 1].value = input_value

        # Process each node (assuming nodes are sorted by rank)
        for node in self.get_hidden_nodes() + self.get_output_nodes():
            total_input = sum(
                conn.weight * self.nodes[conn.in_node.ID].value for conn in node.incoming_connections.values() if
                conn.is_enabled)
            node.value = sigmoid(total_input)

        # Gather output values
        outputs = [node.value for node in self.get_output_nodes()]
        return outputs

    def add_connection_mutation(self):
        total_attempts = 100
        for attempt in range(total_attempts):
            node1, node2 = random.sample(list(self.nodes.values()), 2)
            is_valid_connection, nodes = self.is_valid_connection(node1, node2)
            if is_valid_connection:
                weight = np.random.uniform(-0.5, 0.5)
                self.add_connection(in_node=nodes[0], out_node=nodes[1], weight=weight)
                return True
        return False

    def copy(self):
        new_genome = self.neat.generate_empty_genome(include_in_global_population=False)
        for connection_gene in self.connection_genes.values():
            in_node = connection_gene.in_node
            out_node = connection_gene.out_node

            new_in_node = new_genome.add_node(node_id=in_node.ID, node_type=in_node.type)
            new_out_node = new_genome.add_node(node_id=out_node.ID, node_type=out_node.type)

            new_connection = new_genome.add_connection(in_node=new_in_node, out_node=new_out_node)

            new_connection.is_enabled = connection_gene.is_enabled
            new_connection.weight = connection_gene.weight
            new_genome.connection_genes[new_connection.innovation_number] = new_connection

        for node in self.nodes.values():
            if node.ID in new_genome.nodes:
                continue
            hidden_node = new_genome.add_node(node_id=node.ID, node_type=node.type)
            hidden_node.incoming_connections = node.incoming_connections
            hidden_node.outgoing_connections = node.outgoing_connections

        return new_genome

    def mutate(self):
        if random.random() < self.neat.add_node_probability:
            self.add_node_mutation()
        if random.random() < self.neat.add_connection_probability:
            self.add_connection_mutation()

    def choose_enabled_connection(self):
        enabled_connections = [conn for conn in self.connection_genes.values() if conn.is_enabled]
        if not enabled_connections:
            return None
        return random.choice(enabled_connections)

    def add_node_mutation(self):
        connection_to_split = self.choose_enabled_connection()
        if connection_to_split is None:
            return None
        connection_to_split.is_enabled = False
        innovation = self.neat.innovation_database.get_innovation(connection_to_split.in_node.ID,
                                                                  connection_to_split.out_node.ID)
        if innovation is not None:
            hidden_node = self.add_node(node_id=1000 + innovation.innovation_id, node_type='hidden')

            # else:
            #     new_node_id = self.neat.get_new_node_id()
            #     hidden_node = self.add_node(node_id=1000 + new_node_id, node_type='hidden')
            #
            #     self.neat.innovation_database.add_innovation(innovation_type='Node',
            #                                                  in_node_id=connection_to_split.in_node.ID,
            #                                                  out_node_id=connection_to_split.out_node.ID,
            #                                                  new_node_id=new_node_id)

            in_node_to_hidden_connection = self.add_connection(connection_to_split.in_node, hidden_node)
            hidden_node_to_out_node_connection = self.add_connection(hidden_node, connection_to_split.out_node)

            in_node_to_hidden_connection.weight = 1
            hidden_node_to_out_node_connection.weight = connection_to_split.weight
            return hidden_node

    def add_node(self, node_id, node_type):
        if node_id in self.nodes:
            return self.nodes[node_id]
        node = Node(self, ID=node_id, node_type=node_type)
        self.nodes[node.ID] = node
        return node

    def add_connection(self, in_node, out_node, weight=np.random.randn()):
        connection = self.neat.get_connection(in_node, out_node, weight)
        self.connection_genes[connection.innovation_number] = connection
        in_node.add_outgoing_connection(connection)
        out_node.add_incoming_connection(connection)
        return connection
if __name__ =='__main__':
    print(sigmoid(0.65247598))