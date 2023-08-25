import random
from InnovationDatabase import InnovationDatabase
from Genome import Genome
from Node import Node
from ConnectionGene import ConnectionGene
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Species import Species

# Define the XOR dataset
XOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_OUTPUTS = [0, 1, 1, 0]


def tournament_selection(population, tournament_size):
    # Randomly select `tournament_size` genomes from the population
    tournament = random.sample(population, tournament_size)

    # Return the genome with the highest fitness
    return max(tournament, key=lambda genome: genome.shared_fitness)


# def visualize_genome(genome,generation):
#     G = nx.DiGraph()
#     for node_id, node in genome.nodes.items():
#         G.add_node(node_id)
#     for conn_id, conn in genome.connection_genes.items():
#         if conn.is_enabled:
#             G.add_edge(conn.in_node.ID, conn.out_node.ID, weight=round(conn.weight,2))
#     pos = nx.spring_layout(G)
#     plt.title('Generation {}. Fitness: {}'.format( generation,genome.fitness))
#     nx.draw(G, pos, with_labels=True)
#     labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#     data = []
#     for key, obj in genome.connection_genes.items():
#         connection_gene_data = {'Innovation':obj.innovation_number,
#                                 'in_node': obj.in_node.ID,
#                                 'out_node': obj.out_node.ID,
#                                 'is_enabled': obj.is_enabled}
#         data.append(connection_gene_data)
#
#     df = pd.DataFrame(data)
#     print(df)
#     print()
#
#     plt.show()

class NEAT:
    def __init__(self,
                 total_population=10,
                 total_input_nodes=0,
                 total_output_nodes=0,
                 add_connection_probability=0.2,
                 add_node_probability=0.05,
                 crossover_probability=.7,
                 mutation_probability=.02,
                 elitism_percentage=.2,
                 total_generations=100,
                 tournament_size=5,
                 species_threshold=3,
                 c1=1,
                 c2=1,
                 c3=1
                 ):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.innovation_number = 0
        self.species_id_counter = 0
        self.genome_id_counter = 0
        self.connection_id_counter = 0
        self.total_generations = total_generations
        self.elitism_percentage = elitism_percentage
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.species_threshold = species_threshold
        self.total_population = total_population
        self.total_input_nodes = total_input_nodes
        self.total_output_nodes = total_output_nodes
        self.total_species_created = 0
        self.total_hidden_nodes = 0
        self.total_nodes = total_input_nodes + total_output_nodes

        self.add_connection_probability = add_connection_probability
        self.add_node_probability = add_node_probability
        self.list_of_connection_genes = {}
        self.list_of_Nodes = {}
        self.list_of_Genomes = {}
        self.list_of_Species = {}
        self.innovation_database = InnovationDatabase()

    # Define the fitness function
    def evaluate_xor(self, genome):
        total_error = 0.0
        for xor_input, xor_output in zip(XOR_INPUTS, XOR_OUTPUTS):
            predicted_output = genome.activate(xor_input)
            error = abs(predicted_output[0] - xor_output)
            total_error += error
        # Convert error into fitness
        fitness = len(XOR_OUTPUTS) - total_error
        return fitness

    def get_new_species_id(self):
        self.species_id_counter += 1
        return self.species_id_counter

    def get_new_genome_id(self):
        self.genome_id_counter += 1
        return self.genome_id_counter

    def get_new_connection_id(self):
        self.connection_id_counter += 1
        return self.connection_id_counter

    def generate_empty_genome(self, include_bias=True, include_in_global_population=True):
        genome = Genome(genome_id=self.get_new_genome_id(), neat=self)
        starting_idx_id = 0
        if not include_bias:
            self.initialize_starting_nodes(genome)
        else:
            genome.add_node(node_id=starting_idx_id, node_type='bias')
            self.initialize_starting_nodes(genome)

        if include_in_global_population:
            self.list_of_Genomes[genome.ID] = genome

        return genome

    def get_connection(self, in_node, out_node, weight=np.random.randn()):
        innovation = self.innovation_database.get_innovation(in_node.ID, out_node.ID)
        if innovation is not None:
            return ConnectionGene(innovation.innovation_id, in_node, out_node, weight=weight)
        else:
            innovation = self.innovation_database.add_innovation(innovation_type='Connection',
                                                                 in_node_id=in_node.ID,
                                                                 out_node_id=out_node.ID)
            return ConnectionGene(innovation.innovation_id, in_node, out_node, weight=weight)

    def initialize_starting_nodes(self, genome):
        id_idx = 1
        for n_id in range(self.total_input_nodes):
            node = genome.add_node(node_id=id_idx, node_type='input')
            self.list_of_Nodes[node.ID] = node
            id_idx += 1

        for n_id in range(self.total_output_nodes):
            node = genome.add_node(node_id=id_idx, node_type='output')
            self.list_of_Nodes[node.ID] = node
            id_idx += 1

    def get_total_disjoint_and_excess_genes(self, parent1, parent2):
        parent1_idx = 0
        parent2_idx = 0
        disjoint_genes = 0
        excess_genes = 0
        matching_genes = 0
        total_weight_difference = 0
        average_weight_difference = 0
        parent1_genes = sorted(parent1.get_enabled_connections().values(),
                               key=lambda connection_gene: connection_gene.innovation_number)
        parent2_genes = sorted(parent2.get_enabled_connections().values(),
                               key=lambda connection_gene: connection_gene.innovation_number)

        if len(parent1.connection_genes) == 0 or len(parent2.connection_genes) == 0:
            excess_genes = max(len(parent1.connection_genes), len(parent2.connection_genes))
        else:
            while parent1_idx < len(parent1.connection_genes) and parent2_idx < len(parent2.connection_genes):

                if parent1_idx >= len(parent1_genes):
                    # All genes from parent1 have been processed, remaining genes in parent2 are excess
                    excess_genes += len(parent2_genes) - parent2_idx
                    break
                elif parent2_idx >= len(parent2_genes):
                    # All genes from parent2 have been processed, remaining genes in parent1 are excess
                    excess_genes += len(parent1_genes) - parent1_idx
                    break

                innov1 = parent1_genes[parent1_idx].innovation_number
                innov2 = parent2_genes[parent2_idx].innovation_number
                if innov1 == innov2:
                    total_weight_difference += abs(parent1_genes[parent1_idx].weight - parent2_genes[parent2_idx].weight)
                    parent1_idx += 1
                    parent2_idx += 1
                    matching_genes += 1

                elif innov1 < innov2:
                    parent1_idx += 1
                    disjoint_genes += 1
                else:
                    parent2_idx += 1
                    disjoint_genes += 1

            if matching_genes > 0:
                average_weight_difference = total_weight_difference / matching_genes

        return disjoint_genes, excess_genes, average_weight_difference

    def crossover(self, parent1, parent2):
        if parent2.shared_fitness > parent1.shared_fitness:
            parent1, parent2 = parent2, parent1

        child_genome = self.generate_empty_genome(include_bias=True, include_in_global_population=False)

        # Get the union of innovation numbers from both parents
        all_innovations = set(parent1.connection_genes.keys()).union(set(parent2.connection_genes.keys()))

        for innovation_number in sorted(all_innovations):
            # Check if the innovation number exists in both parents
            if innovation_number in parent1.connection_genes and innovation_number in parent2.connection_genes:
                # Choose a parent based on a random probability
                chosen_parent = parent1 if random.random() < 0.5 else parent2

                # Extract in_node and out_node from the chosen parent
                in_node_id = chosen_parent.connection_genes[innovation_number].in_node.ID
                out_node_id = chosen_parent.connection_genes[innovation_number].out_node.ID

                in_node_type = chosen_parent.connection_genes[innovation_number].in_node.type
                out_node_type = chosen_parent.connection_genes[innovation_number].out_node.type

                # Add or retrieve nodes for the child genome
                in_node = child_genome.nodes.get(in_node_id) or child_genome.add_node(node_id=in_node_id,
                                                                                      node_type=in_node_type)
                out_node = child_genome.nodes.get(out_node_id) or child_genome.add_node(node_id=out_node_id,
                                                                                        node_type=out_node_type)
                # Add connection for the child genome
                child_inherited_connection_gene = child_genome.add_connection(in_node, out_node)
                child_inherited_connection_gene.weight = chosen_parent.connection_genes[innovation_number].weight
                child_inherited_connection_gene.is_enabled = chosen_parent.connection_genes[
                    innovation_number].is_enabled
                continue

            chosen_parent = None
            if innovation_number in parent1.connection_genes:
                chosen_parent = parent1
            elif innovation_number in parent2.connection_genes and parent1.shared_fitness == parent2.shared_fitness:
                chosen_parent = parent2

            if chosen_parent:
                chosen_parent_in_node = chosen_parent.connection_genes[innovation_number].in_node
                chosen_parent_out_node = chosen_parent.connection_genes[innovation_number].out_node

                in_node = child_genome.nodes.get(chosen_parent_in_node.ID) or child_genome.add_node(
                    node_id=chosen_parent_in_node.ID,
                    node_type=chosen_parent_in_node.type)

                out_node = child_genome.nodes.get(chosen_parent_out_node.ID) or child_genome.add_node(
                    node_id=chosen_parent_out_node.ID,
                    node_type=chosen_parent_out_node.type)

                child_inherited_connection_gene = child_genome.add_connection(in_node, out_node)
                child_inherited_connection_gene.weight = chosen_parent.connection_genes[innovation_number].weight
                child_inherited_connection_gene.is_enabled = chosen_parent.connection_genes[
                    innovation_number].is_enabled

        # Ensure all nodes corresponding to connection genes are in the child genome
        for connection in child_genome.connection_genes.values():
            in_node = connection.in_node
            out_node = connection.out_node
            if in_node.ID not in child_genome.nodes:
                child_genome.nodes[in_node.ID] = in_node
            if out_node.ID not in child_genome.nodes:
                child_genome.nodes[out_node.ID] = out_node

        return child_genome

    def initialize_starting_species(self):
        initial_species = Species(ID=self.get_new_species_id(), neat=self)
        for genome in self.list_of_Genomes.values():
            initial_species.add_member(genome)
        initial_species.initialize_representative()
        self.list_of_Species[initial_species.ID] = initial_species

    def initialize_starting_population(self):
        for pop in range(self.total_population):
            self.generate_empty_genome(include_bias=True)
        return self.list_of_Genomes

    def evaluate_fitness(self):
        for genome in self.list_of_Genomes.values():
            fitness = self.evaluate_xor(genome)
            genome.fitness = fitness

    def reassign_species(self):
        for species in self.list_of_Species.values():
            species.clear_members()

        for genome in self.list_of_Genomes.values():
            assigned = False
            for species in self.list_of_Species.values():
                compatibility_distance = species.calculate_compatibility(genome)
                if compatibility_distance < self.species_threshold:
                    species.add_member(genome)
                    assigned = True
                    break
            if not assigned:
                new_species = self.generate_new_species()
                new_species.add_member(genome)
                new_species.initialize_representative()
                self.list_of_Species[new_species.ID] = new_species

    def generate_new_species(self):
        species = Species(ID=self.get_new_species_id(), neat=self)
        self.list_of_Species[species.ID] = species
        return species

    def calculate_shared_fitness(self):
        for species in self.list_of_Species.values():
            for genome in species.list_of_members.values():
                raw_fitness = genome.fitness
                compatibility_sum = sum(self.calculate_compatibility(genome, other_genome) for other_genome in
                                        species.list_of_members.values() if genome != other_genome)
                shared_fitness = raw_fitness / (1 + compatibility_sum)
                genome.shared_fitness = shared_fitness

    def calculate_compatibility(self, genome1, genome2):
        total_excess, total_disjoint, average_weight_difference = self.get_total_disjoint_and_excess_genes(genome1,
                                                                                                           genome2)
        N = max(len(genome1.get_enabled_connections()), len(genome2.get_enabled_connections()))
        if N < 20:
            N = 1
        distance = (self.c1 * total_excess) / N + (self.c2 * total_disjoint) / N + (
                self.c3 * average_weight_difference)
        return distance

    def generate_elites(self):
        sorted_genomes = sorted(self.list_of_Genomes.values(), key=lambda genome: genome.shared_fitness,
                                reverse=True)
        num_elites = int(self.elitism_percentage * len(sorted_genomes))
        elites = sorted_genomes[:num_elites]
        new_genomes = elites[:]
        return new_genomes

    def evolve(self):
        crossover_prob = self.crossover_probability
        mutations_prob = self.mutation_probability
        generations = self.total_generations
        tournament_size = self.tournament_size
        self.initialize_starting_population()
        self.initialize_starting_species()
        for generation in range(generations):
            # print('GENERATION#', generation)
            # print()
            self.evaluate_fitness()
            self.calculate_shared_fitness()
            new_genomes = self.generate_elites()
            # Continue with selection, crossover, and mutation to fill the rest of the new generation
            while len(new_genomes) < len(self.list_of_Genomes):
                parent1 = tournament_selection(list(self.list_of_Genomes.values()), tournament_size)
                parent2 = tournament_selection(list(self.list_of_Genomes.values()), tournament_size)

                if random.random() < crossover_prob and parent1 != parent2:
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2]).copy()

                if random.random() < mutations_prob:
                    child.mutate()  # Assuming you have a mutate method for genomes
                new_genomes.append(child)

            self.list_of_Genomes = {genome.ID: genome for genome in new_genomes}

            self.reassign_species()
            # Print stats for the generation
            best_fitness = max([genome.shared_fitness for genome in self.list_of_Genomes.values()])
            # print(f"Pop size:", len(self.list_of_Genomes.values()))
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            print('Total Innovations', len(self.innovation_database.innovations))
            # print()



if __name__ == '__main__':
    neat = NEAT(total_input_nodes=2,
                crossover_probability=.7,
                mutation_probability=.8,
                tournament_size=2,
                add_node_probability=.05,
                add_connection_probability=.8,
                total_output_nodes=1,
                total_population=150,
                total_generations=100)
    neat.evolve()
