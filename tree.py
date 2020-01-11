from reversi import *
import config


class Node:
    def __init__(self, board, tern, parent):
        # self.N = 1 # visit times 
        # self.W = 0
        # self.Q = 0
        self.visit_times = 1
        self.total_value = 0
        self.parent = parent
        self.children = []
        self.board = board
        self.tern = tern
        self.available_pos = available_pos(self.board, self.tern)
        if len(self.available_pos) == 0:
            self.tern = -self.tern
            self.available_pos = available_pos(self.board, self.tern)
            if len(available_pos(self.board, self.tern)) == 0:
                self.tern = 0
        


class Tree:
    def __init__(self):
        self.node_ptr = dict()
    def create_root(self, board, tern):
        self.root = Node(board, tern, None)
    
