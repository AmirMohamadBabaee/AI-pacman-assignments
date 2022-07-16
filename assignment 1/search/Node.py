# Node.py

# This file has implementation of Node class to use to create State Graph
# of Pacman. 

class Node:

    def __init__(self, current_state, parent, parent_dir, cost=0) -> None:
        self.current_state = current_state
        self.parent = parent
        self.parent_dir = parent_dir  # the direction that we come from to this Node
        self.north = None
        self.east = None
        self.west = None
        self.south = None
        self.cost = cost

    # Set a parent state to this state
    def set_parent(self, parent, parent_dir):
        self.parent = parent
        self.parent_dir = parent_dir

    # Set a child to this state
    def set_child(self, direction, child_state):
        if direction == 'North':
            self.north = child_state
        elif direction == 'East':
            self.east = child_state
        elif direction == 'West':
            self.west = child_state
        else:
            self.south = child_state