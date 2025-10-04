############################################################
# CMPSC 442: Informed Search
############################################################

student_name = "Aidan Vesci"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.



############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    pass

class TilePuzzle(object):
    
    # Required
    def __init__(self, board):
        pass

    def get_board(self):
        pass

    def perform_move(self, direction):
        pass

    def scramble(self, num_moves):
        pass

    def is_solved(self):
        pass

    def copy(self):
        pass

    def successors(self):
        pass

    # Required
    def get_solutions_iddfs(self):
        pass

    # Required
    def get_solution_A_star(self):
        pass
    
    
############################################################
# Section 2: Grid Navigation
############################################################

def get_neighbors(node, scene):
    pass

def find_shortest_path(start, goal, scene):
    pass


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

def heuristic(state, goal):
    pass

def solve_distinct_disks_v2(length, n):
    pass

############################################################
# Section 4: Dominoes Game
############################################################

def make_dominoes_game(rows, cols):
    pass

class DominoesGame(object):

    # Required
    def __init__(self, board):
        pass

    def get_board(self):
        pass

    def reset(self):
        pass

    def is_legal_move(self, row, col, vertical):
        pass

    def legal_moves(self, vertical):
        pass

    def execute_move(self, row, col, vertical):
        pass

    def game_over(self, vertical):
        pass

    def copy(self):
        pass

    def successors(self, vertical):
        pass

    def get_random_move(self, vertical):
        pass

    # Required
    def get_best_move(self, limit, vertical):
        pass

