import unittest
from NEAT import NEAT
import numpy as np
class TestNEAT(unittest.TestCase):

    def test_initialization(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        has_bias = True
        genome = neat.generate_empty_genome(include_bias=has_bias, include_in_global_population=False)
        if has_bias:  # has total inputs and outputs
            self.assertEqual(len(genome.nodes), neat.total_input_nodes + neat.total_output_nodes + 1)
        else:
            self.assertEqual(len(genome.nodes), neat.total_input_nodes + neat.total_output_nodes)
        initial_population = neat.initialize_starting_population()
        self.assertEqual(neat.total_population, len(initial_population))

    def test_activate(self):
        # Create a NEAT object
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)

        # Create an empty genome
        genome = neat.generate_empty_genome(include_bias=True, include_in_global_population=False)

        # Add nodes and connections to represent the XOR problem
        # Assuming you have methods to add nodes and connections
        # You'll need to replace these with the actual code to set up the genome

        genome.add_connection(in_node=genome.nodes[0], out_node=genome.nodes[3], weight=np.random.uniform(-0.5, 0.5))  # Bias to output
        genome.add_connection(in_node=genome.nodes[1], out_node=genome.nodes[3], weight=np.random.uniform(-0.5, 0.5))  # Input1 to output
        genome.add_connection(in_node=genome.nodes[2], out_node=genome.nodes[3], weight=np.random.uniform(-0.5, 0.5))  # Input2 to output
        # Add hidden nodes and connections as needed

        # Define the inputs and expected outputs for the XOR function
        test_cases = [
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        ]

        # Tolerance for checking the output
        tolerance = 0.1

        for inputs, expected_output in test_cases:
            # Call the activate function with the inputs
            outputs = genome.activate(inputs)

            # Check that the output is close to the expected value
            assert abs(outputs[0] - expected_output[
                0]) < tolerance, f"Failed for inputs {inputs}: got {outputs[0]}, expected {expected_output[0]}"

    def test_initialize_starting_species(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        neat.initialize_starting_population()
        initial_total_species = len(neat.list_of_Species)

        for genome in neat.list_of_Genomes.values():
            species = genome.species
            assert species is None

        neat.initialize_starting_species()

        for genome in neat.list_of_Genomes.values():
            species = genome.species
            assert species is not None

        self.assertEqual(initial_total_species + 1, len(neat.list_of_Species))
        representative = neat.list_of_Species[1].representative
        assert representative is not None


    def test_get_total_disjoint_and_excess_genes(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        parent1 = neat.generate_empty_genome()
        parent2 = neat.generate_empty_genome()
        parent1.add_connection_mutation()
        parent1.add_connection_mutation()
        parent2.add_connection_mutation()
        parent2.add_node_mutation()
        parent2.add_node_mutation()
        disjoint, excess, average_weight_difference = neat.get_total_disjoint_and_excess_genes(parent1, parent2)
        self.assertEqual(2, disjoint)
        self.assertEqual(3, excess)
        print('disjoint',disjoint)
        print('excess',excess)

    def test_evaluate_fitness(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1, total_population=10, total_generations=100)
        neat.initialize_starting_population()
        neat.evaluate_fitness()
        for genome in neat.list_of_Genomes.values():
            self.assertGreater( genome.fitness, 0)

    def test_calculate_shared_fitness(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1, total_population=10, total_generations=100)
        neat.initialize_starting_population()
        neat.initialize_starting_species()
        neat.evaluate_fitness()
        neat.calculate_shared_fitness()
        for genome in neat.list_of_Genomes.values():
            self.assertGreater( genome.shared_fitness, 0)



    def test_evolve(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1, total_population=10, total_generations=500)
        neat.evolve()
        self.assertEqual(neat.total_population, len(neat.list_of_Genomes))


    def test_reassign_species(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1, total_population=10, total_generations=500)
        neat.initialize_starting_population()
        neat.initialize_starting_species()
        neat.evaluate_fitness()
        neat.calculate_shared_fitness()
        neat.reassign_species()



    def test_generate_elites(self):
        self.assertTrue(True)



    def test_evaluate_xor(self):
        XOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
        XOR_OUTPUTS = [0, 1, 1, 0]
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        initial_population = neat.initialize_starting_population()
        genome = initial_population[1]
        genome.add_connection_mutation()
        genome.add_node_mutation()
        neat.evaluate_xor(genome)






    def test_add_connection_mutation(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        genome = neat.generate_empty_genome()
        initial_connection_count = len(genome.connection_genes)
        genome.add_connection_mutation()
        self.assertEqual(len(genome.connection_genes), initial_connection_count + 1)


    def test_add_node_mutation(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        genome = neat.generate_empty_genome()
        genome.add_connection_mutation()
        initial_node_count = len(genome.nodes)
        genome.add_node_mutation()
        self.assertEqual(len(genome.nodes), initial_node_count + 1)

    def test_crossover(self):
        neat = NEAT(total_input_nodes=2, total_output_nodes=1)
        parent1 = neat.generate_empty_genome()
        parent1.connection_genes[1] = neat.get_connection(parent1.nodes[1], parent1.nodes[3])
        parent1.connection_genes[2] = neat.get_connection(parent1.nodes[2], parent1.nodes[3])
        parent1.connection_genes[3] = neat.get_connection(parent1.nodes[0], parent1.nodes[3])

        parent2 = neat.generate_empty_genome()
        parent2.connection_genes[2] = neat.get_connection(parent2.nodes[2], parent2.nodes[3])
        parent2.connection_genes[3] = neat.get_connection(parent2.nodes[0], parent2.nodes[3])
        hidden_node = parent2.add_node(node_id=1000 + 4, node_type='hidden')
        parent2.connection_genes[4] = neat.get_connection(parent2.nodes[2], hidden_node)
        parent2.connection_genes[5] = neat.get_connection(hidden_node, parent2.nodes[3])
        parent2.connection_genes[6] = neat.get_connection(parent2.nodes[1], hidden_node)

        child_genome = neat.crossover(parent1, parent2)
        expected_connections = [(1, 3), (2, 3), (0, 3), (2, 1004), (1004, 3), (1, 1004)]
        self.assertEqual(len(expected_connections), len(child_genome.connection_genes))
        self.assertEqual(5, len(child_genome.nodes))

        for i in child_genome.connection_genes.values():
            in_node_ID = i.in_node.ID
            out_node_ID = i.out_node.ID
            pair = (in_node_ID, out_node_ID)
            self.assertTrue(pair in expected_connections)

        # Check output connections
        output_node = child_genome.nodes[3]
        self.assertEqual(4, len(output_node.incoming_connections))

        hidden_node = child_genome.nodes[1004]
        self.assertEqual(2, len(hidden_node.incoming_connections))
        self.assertEqual(1, len(hidden_node.outgoing_connections))

        # Check input connections
        input_node_1 = child_genome.nodes[1]
        self.assertEqual(2, len(input_node_1.outgoing_connections))
        input_node_2 = child_genome.nodes[1]
        self.assertEqual(2, len(input_node_2.outgoing_connections))
        input_node_0 = child_genome.nodes[0]
        self.assertEqual(1, len(input_node_0.outgoing_connections))


if __name__ == '__main__':
    unittest.main()
