import datetime


class Innovation:
    def __init__(self, innovation_id, innovation_type, in_node_id, out_node_id, new_node_id=None):
        self.innovation_id = innovation_id
        self.innovation_type = innovation_type
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.new_node_id = new_node_id
        self.timestamp = datetime.datetime.now()


class InnovationDatabase:
    def __init__(self):
        self.innovations = []
        self.current_id = 0

    def add_innovation(self, innovation_type, in_node_id, out_node_id, new_node_id=None):
        self.current_id += 1
        innovation = Innovation(self.current_id, innovation_type, in_node_id, out_node_id, new_node_id)
        self.innovations.append(innovation)
        return innovation

    def get_innovation(self, in_node_id, out_node_id):
        for innovation in self.innovations:
            if innovation.in_node_id == in_node_id and innovation.out_node_id == out_node_id:
                return innovation
        return None

