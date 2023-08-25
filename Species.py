import random


class Species:
    def __init__(self, ID, neat):
        self.ID = ID
        self.neat = neat
        self.fitness = 0.0
        self.representative = None
        self.list_of_members = {}

    def add_member(self, genome):
        self.list_of_members[genome.ID] = genome
        genome.species = self

    def initialize_representative(self):
        self.representative = random.sample(list(self.list_of_members.values()), 1)[0]

    def get_representative(self):
        return self.representative

    def calculate_compatibility(self, genome):
        total_disjoint, total_excess, average_weight_difference = self.neat.get_total_disjoint_and_excess_genes(genome, self.get_representative())
        N = max(len(genome.get_enabled_connections()), len(self.representative.get_enabled_connections()))
        if N < 20:
            N = 1
        distance = (self.neat.c1 * total_excess) / N + (self.neat.c2 * total_disjoint) / N + (self.neat.c3 * average_weight_difference)
        return distance

    def clear_members(self):
        for genome in self.list_of_members.values():
            genome.species = None
        self.list_of_members = {}
