class Node:
    def __init__(self, neat, ID, node_type):
        rank = {"bias": 1, "input": 1, "hidden": 2, "output": 3}
        self.neat = neat
        self.value = 0.0
        self.ID = ID
        self.type = node_type
        self.rank = rank[node_type]
        self.incoming_connections = {}
        self.outgoing_connections = {}

    def add_incoming_connection(self, connection):
        self.incoming_connections[connection.innovation_number] = connection

    def add_outgoing_connection(self, connection):
        self.outgoing_connections[connection.innovation_number] = connection
