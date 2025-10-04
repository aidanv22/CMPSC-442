############################################################
# CMPSC 442: Informed Search
############################################################

student_name = "Aidan Vesci"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random
from copy import deepcopy
from queue import PriorityQueue
import math

############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    # Build the board row by row
    board = [] # create board that will be a 2D list
    tile_value = 0 # start with the first tile in top left being 0 matching the goal state
    for row in range(rows): # loop over rows
        row_list = [] # create a new list for each row
        for column in range(cols): # loop over columns
            row_list.append(tile_value) # place current tile value into row at current column
            tile_value += 1 # move to next tile number
        board.append(row_list) # add completed row to overarching board

    return TilePuzzle(board)

class TilePuzzle(object):
    
    # Required
    def __init__(self, board):
        # same implementation as HW 1 with added code
        self.board = board
        self.row_length = len(board)
        self.column_length = len(board[0])

        # Find the position of the empty tile in init so the search is faster and we don't have to find it every time we want to perform a move
        # below we check every cell
        for row in range(self.row_length): # go through every row
            for col in range(self.column_length): # and every column
                if board[row][col] == 0: # if the cell we are on now is empty add it to empty tile
                    self.empty_tile = (row, col)
                    break # break out of loop after it is found

        # Precompute goal positions for where each tile should be when board is solved
        self.tile_goals = {} # initialize a dictionary to check later if puzzle is solved
        counter = 0 # top left corner starts with zero
        for row in range(self.row_length): # go through every row
            for col in range(self.column_length): # and go through every column
                self.tile_goals[counter] = (row, col) # assigns every number to its goal position
                counter += 1

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        # set the current blank location from the empty tile tuple we saved earlier
        row, column = self.empty_tile

        # computing move direction 

        if direction == "up":
            new_row, new_column = row - 1, column # if move is up we subtract 1 from row, since row 0 is top and row 1 below, moving up subtracts a row

        elif direction == "down":
            new_row, new_column = row + 1, column # if move is down we add 1 from row, since row 0 is top and row 1 below, moving down adds a row

        elif direction == "left":
            new_row, new_column = row, column - 1 # if move is left we subtract 1 from col, since col 0 is left and col 1 is more right, moving up subtracts a column

        elif direction == "right":
            new_row, new_column = row, column + 1 # if move is left we subtract add from col, since col 0 is left and col 1 is more right, moving right adds a column
        else:
            return False # if the direction string is invalid

        # if the new position we try to move to is less then 0 or outside the board legnth we return false
        if new_row < 0 or new_row >= self.row_length or new_column < 0 or new_column >= self.column_length:
            return False
    
        # Swap the empty tile with the neighbor
        temp = self.board[row][column]
        self.board[row][column] = self.board[new_row][new_column]
        self.board[new_row][new_column] = temp


        # Update the empty tiles position
        self.empty_tile = (new_row, new_column)

        return True # return True is the move was succesful / valid

    def scramble(self, num_moves):
         allowed_moves = ["up", "down", "left", "right"] # list of allowed moves
         for move in range(num_moves): # for however many random moves
            # Pick a random direction
            direction = random.choice(allowed_moves) # perform a move at random

            # Try to move in that direction
            try_move = self.perform_move(direction) # try to perform the move 

            # if a move is invalid skip the move and keep going
            if not try_move:
                continue

    def is_solved(self):
        counter = 0 # start with zero because goal state has 0 in top left
        for row in range(self.row_length): # go through every row
            for col in range(self.column_length): # and go through every column
                if self.board[row][col] == counter: # if the row,column has the correct tile value keep going
                    counter +=1
                else: # if number doesnt match return False
                    return False
        return True


    '''
    create a deepcopy so that changes made in the original puzzle aren't reflected in the copy and visa versa
    '''
    def copy(self):
        return TilePuzzle(deepcopy(self.board)) 
    
    def successors(self):
        for move_made in ("up", "down", "left", "right"): # iterate over the four possible moves
            newTilePuzzle = self.copy() # create a copy of current puzzle before attempting move
            if not newTilePuzzle.perform_move(move_made): # if the move is invalid skip and continue
                continue
            else:
                yield move_made, newTilePuzzle # otherwise yield a tuple with the move made and the new puzzle

    # Required
    def get_solutions_iddfs(self):
        depth = 0 # start with 0 for iterative deepening
        found = False # start off with no solution
        while not found: # while a solution hasn't been found keep looping
            '''
            call the recursive helper function with the current depth
            we  create a move list with an arbitrary move and the intitial puzzle
            '''
            for solution in self._iddfs_helper(depth, [(None, self)]): # call the recursive helper function with the current depth.
                found = True # if we have found a solution mark it true so the loop stops
                yield solution # and yield the solution
            if found: # if the solution was found leave the loop
                break
            else: # otherwise increase the depth and run the proccess again
                depth += 1

    def _iddfs_helper(self, limit, moves):
        recent_move, current_puzzle = moves[-1] # get the most recent move and puzzle state from the list of move


        if current_puzzle.is_solved(): # if the current puzzle is solved (blank in top left and numbers in order)
            yield [move[0] for move in moves[1:]] # yield the moves taken / path to get to that solution excluding the dummy move we put in first
            return # return to stop recursion

        if limit == 0: # if you've reached depth limit stop exploring
            return

        for current_move, next_puzzle in current_puzzle.successors(): # iterate over all possible moves from current puzzle

            '''
            Not sure if this is needed, but we want to skip moves that are the reverse of what we just did
            This way we can avoid redundant puzzle states and only search what we want
            '''
            if (recent_move == "up" and current_move == "down") or (recent_move == "down" and current_move == "up") or \
               (recent_move == "left" and current_move == "right") or (recent_move == "right" and current_move == "left"):
                continue

            new_moves = moves + [(current_move, next_puzzle)] # add new move and new puzzle state to path
            yield from self._iddfs_helper(limit - 1, new_moves) # recursively explore the new state
  
        

    # Required
    def get_solution_A_star(self):
        frontier = PriorityQueue() # intialize priority queue to hold puzzle states by cost of f(n) = g(n) + h(n)
        visited_configs = set() # set to track board configurations already visited
        tie_breaker = 0  # implemented this because python will raise an error, but if multiple states have the same value we can ensure proper ordering

        initial_h = self.heuristic_helper() # use the helper function to create  a manhattan distance for the initial puzzle state
        frontier.put((initial_h, tie_breaker, self.copy(), [])) # we add the initial puzzle to the queue with the intial heuristic cost, tie breaker, current board state, and the path so far

        while not frontier.empty(): # while the queue still has elements
            f_cost, _, current_board, move_path = frontier.get() # get the puzzle state with the lowest cost
            current_board_state = tuple(tuple(row) for row in current_board.board) # converting the current board to tuple so we can store it in visited and avoid going back to already explored boards
            
            # skip the board state if we've already seen it
            if current_board_state in visited_configs:
                continue
            else: # otherwise add the board state to set so we don't repeat its exploration
                visited_configs.add(current_board_state)
            
            # if the board is solved return path it took to get there
            if current_board.is_solved():
                return move_path

            '''
            loop over all the  valid moves from the current puzzle state -> 
                current move is ["up", "down", etc.]; 
                next_puzzle is the resulting puzzle after move is made
            this is A* in a nutshell, expanding by generating children

            '''
            for current_move, next_puzzle in current_board.successors():
                next_board_state = tuple(tuple(row) for row in next_puzzle.board) # converting the next/resulting board to tuple so we can store it in visited and avoid going back to already explored boards
                if next_board_state in visited_configs: # skip board configuration / state if we've already seen it
                    continue

                g_n = len(move_path) + 1 # g(n) is the cost so far to reach the resulting puzzle and we add 1 because we need to account for the next move we are about to take
                h_n = next_puzzle.heuristic_helper() # h(n) is the estimated cost to reach the goal state from the resulting puzzle; we use the manhattan distance here to help
                f_n = g_n + h_n # f is the total estimated cost
                tie_breaker += 1 # add one to the tie-breaker to ensure unique ordering in the priority queue and avoid python throwing an error
                frontier.put((f_n, tie_breaker, next_puzzle, move_path + [current_move])) # we add the succesing puzzle to the queue with the total cost, tie breaker, new board / puzzle state, and the updated path with all moves taken to get to this point

    def heuristic_helper(self):
        manhattan_dist = 0 # total manhattan distance so far
        for row in range(self.row_length): # loop over every row
            for column in range(self.column_length): # and every column to see every tile
                val = self.board[row][column] # we get the value of the tile at this position of the loop
                if val == 0: 
                    goal_row, goal_column = 0, 0  # we need the blank tile to be top left in goal state
                else:
                    # otherwise we compute the goal tile position for the tile 
                    goal_row = val // self.column_length
                    goal_column = val % self.column_length
                manhattan_dist += abs(row - goal_row) + abs(column - goal_column) # add the manhattan distance for this tile
        return manhattan_dist # return the heuristic estimate / manhattan distance
    
    
############################################################
# Section 2: Grid Navigation
############################################################

def get_neighbors(node, scene):
    neighbors = [] # list to store neighbors
    rows= len(scene) # number of rows in grid
    columns = len(scene[0]) # number of columns in grid
    
    '''
    dictionary of directions for lookup:
        values are (row offset, column offset)
        ie. if you want to move up a row you go from row1 -> row0 so you subtract 1
        and so on...
    '''
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
        "up-left": (-1, -1),
        "up-right": (-1, 1),
        "down-left": (1, -1),
        "down-right": (1, 1)
    }

    # loop through every direction label (ie. "up", "down", etc.)
    for direction_name in directions.keys():
        row_direction,column_direction = directions[direction_name] # unpacck the dictionary values into row and column directions
        newRow = row_direction + node[0] # calculate new row index of neighbor by applying the vertical offset to the current row
        newColumn = column_direction + node[1] # calculate new column index of neighbor by applying the horizontal offset to the current column

        if 0 <= newRow < rows and 0 <= newColumn < columns: # if the new position is inside the grid boundaries
            if scene[newRow][newColumn] == True: # we check if the cell at the new row and column we are trying to access is an obstacle
                continue # if it is skip and continue the loop
            else: # otherwise:
                neighbors.append((newRow, newColumn)) # append the valid neighbor to the list
    return neighbors # return the list of all the neighbors


def find_shortest_path(start, goal, scene):
    
    visited_nodes = set() # initialize a set of visited nodes to avoid redundancy
    PQ = PriorityQueue() # intitialie a Priority Queue to explore nodes according to total cost f(n)

    if scene[start[0]][start[1]] or scene[goal[0]][goal[1]]: # if either the start point or goal point are obstacles we cant find a path
        return None


    # we want to compute the estimate heuristic from start point to goal point
    initial_f = euclidean_helper(start, goal)

    '''
    Push the starting node into the Queue where:
        f(n) = g(n) + h(n) -- estimated total cost
        g(n) = cost to this point
        start: start node or at this point current node
        [start]: moves made / path to get here so far
    '''
    PQ.put((initial_f, 0, start, [start]))

    # as long as the Priority Queue still has nodes in it, keep exploring
    while not PQ.empty():
        f_n, g_n, current_node, moves_taken_path = PQ.get() # we pop the node with the lowest total cost / f(n) value according to algorithm

        if current_node == goal: # if the current node we are on is the goal point
            return moves_taken_path # return the path taken

        if current_node in visited_nodes: # if we have already explored / gone through the node we are on
            continue # skip and go to next one
        else: # otherwise:
            visited_nodes.add(current_node) # add it to the set of nodes already explored

        # we loop through all the valid neighbors of our current node using the get neighbors fxn
        for neighbor in get_neighbors(current_node, scene):
            if neighbor not in visited_nodes: # if we havent explored this neighbor before
                new_g_n = g_n + 1  # increment the cost so far (g(n)) by 1
                new_f_n = new_g_n + euclidean_helper(neighbor, goal) # compute the total estimated cost f(n) = g(n) + h(n)
                PQ.put((new_f_n, new_g_n, neighbor, moves_taken_path + [neighbor])) # push the neighbor into the queue with updated: (total estimated cost, cost so far, the neighbor visited, and the path to get to this neighbor)

    return None  # if we don't reach the goal point by the end of the loop (ie. we went through the entire Priority Queue, return none -- we've failed)
    

def euclidean_helper(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) # euclidean distance helper fxn


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def heuristic(state, goal):
    moves_required = 0
    for current_position, goal_position in zip(state,goal): # loop through each disk in the starting state and what its position should be in the goal state
        distance_to_goal = abs(current_position-goal_position) # calculating the distance from the current index of the disk to the goal index; absolute value ensures it is positive
        moves_required += math.ceil(distance_to_goal / 2) # estimating the minimum number of moves, since 2 jumps is the largest move we can round up to get the minimum
    return moves_required # return the number of moves required


def solve_distinct_disks_v2(length, n):
     # if there are no disks to move return an empty list
    if n == 0:
        return []

    # Initial and goal states
    start_state = tuple(list(range(1, n + 1)) + [0] * (length - n)) # leftmost n cells are filled with disks, the rest are filled with zeros
    goal_state = tuple([0] * (length - n) + sorted(range(1, n + 1), reverse=True)) # in the goal state the rightmost n cells are filled with disks in reverse order (D3,D2,D1), and the rest of the left is filled with zeros

    PQ = PriorityQueue() # initialize a Priority Queue to always explore the lowest cost state next
    PQ.put((0, 0, start_state, [])) # add the sarting state to the Priority Queue with: (initial guess -- f(n), move score so far -- g(n), the start state, and move history)
    visited_states = set() # create a set to track visited states and avoid redundancy

    while not PQ.empty(): # while the Priority Queue still has states to explore
        f_n, g_n, current_state, moves_taken = PQ.get() # get the next best state based on the lowest value of f(n)

        # check if the current state is the goal state, if it is return the list of moves used to get there
        if current_state == goal_state:
            return moves_taken

        '''
        this loop iterates over positions of i where a disk could either move: 
        one spot to the right (i+1) 
        or 
        two spots to the right (i+2)
        or
        one spot to the left (i-1)
        or
        two spots to the left (i-2)
        '''
        for i in range(length): # since we can move backwards we need to go to the full length to potenitally check every index

            # RIGHT

            # Single jump of i right: i+1
            if i + 1 < length and current_state[i] != 0 and current_state[i + 1] == 0: # sees if a one step move is legal (in bounds) with a disk at i and an empty space at i+1 (we use state[i] != 0 because of distinct disks)
                new_state = list(current_state) # this tuple gets copied to a list for later mutation
                new_state[i], new_state[i + 1] = 0, current_state[i] # applies the move one step to the right (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple

                if new_state not in visited_states: # avoiding going back over states we've already seen
                    visited_states.add(new_state) # if we havent seen it add it to the set
                    h_n = heuristic(new_state, goal_state) # compute the heuristic element from this new state to goal state
                    PQ.put((g_n + 1 + h_n, g_n + 1, new_state, moves_taken + [(i, i + 1)])) # add the new state to the priority queue with: (the f(n) value, the g(n) value, the new board state / configuration, the path to get to this point)

            # Double jump of i right: i+2 (could also jump over one disk)
            if i + 2 < length and current_state[i] != 0 and current_state[i + 1] != 0 and current_state[i + 2] == 0: # sees if a two step move is legal (index not out of bounds), with a disk at i+1 that the disk at i will jump over and an empty space at i+2 (we use state[i] != 0, state[i+1] != 0  because of distinct disks)
                new_state = list(current_state) # this tuple gets copied to a list for later mutation as shown above
                new_state[i], new_state[i + 2] = 0, current_state[i]  # applies the move  from i to i+2 (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple as shown above

                if new_state not in visited_states: # again avoiding going back over states we've already seen
                    visited_states.add(new_state) # again if we havent seen it add it to the set
                    h_n = heuristic(new_state, goal_state) # compute the heuristic element from this new state to goal state
                    PQ.put((g_n + 1 + h_n, g_n + 1, new_state, moves_taken + [(i, i + 2)])) # add the new state to the priority queue with: (the f(n) value, the g(n) value, the new board state / configuration, the path to get to this point)

            # LEFT

            # Single jump of i left: i+1
            if i - 1 >= 0 and current_state[i] != 0 and current_state[i - 1] == 0: # sees if a one step move is legal (index not out of bounds) with a disk at i and an empty space at i-1 (we use state[i] != 0 because of distinct disks)
                new_state = list(current_state) # this tuple gets copied to a list for later mutation
                new_state[i], new_state[i - 1] = 0, current_state[i] # applies the move  from i to i-1 (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple

                if new_state not in visited_states: # avoiding going back over states we've already seen
                    visited_states.add(new_state) # if we havent seen it add it to the set
                    h_n = heuristic(new_state, goal_state) # compute the heuristic element from this new state to goal state
                    PQ.put((g_n + 1 + h_n, g_n + 1, new_state, moves_taken + [(i, i - 1)])) # add the new state to the priority queue with: (the f(n) value, the g(n) value, the new board state / configuration, the path to get to this point)

            # Double jump of i left: i+2 (could also jump over one disk)
            if i - 2 >= 0 and current_state[i] != 0 and current_state[i - 1] != 0 and current_state[i - 2] == 0: # sees if a two step move is legal (index not out of bounds), with a disk at i-1 that the disk at i will jump over and an empty space at i-2 (we use state[i] != 0, state[i-1] != 0 because of distinct disks)
                new_state = list(current_state) # this tuple gets copied to a list for later mutation as shown above
                new_state[i], new_state[i - 2] = 0, current_state[i]  # applies the move  from i to i-2 (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple as shown above

                if new_state not in visited_states: # again avoiding going back over states we've already seen
                    visited_states.add(new_state) # again if we havent seen it add it to the set
                    h_n = heuristic(new_state, goal_state) # compute the heuristic element from this new state to goal state
                    PQ.put((g_n + 1 + h_n, g_n + 1, new_state, moves_taken + [(i, i - 2)])) # add the new state to the priority queue with: (the f(n) value, the g(n) value, the new board state / configuration, the path to get to this point)

    # if there are no possible moves return None
    return None

############################################################
# Section 4: Dominoes Game
############################################################

def make_dominoes_game(rows, cols):
    board = []  # we start with an empty board
    for row in range(rows):  # we loop through every row
        board.append([])  # and add a new empty row
        for column in range(cols):  # we then loop through every column
            board[row].append(False)  # and set every cell = False
    return DominoesGame(board)  # return a new game with this board

class DominoesGame(object):

    # Required
    def __init__(self, board):
        # same implementatio as Tile Puzzle
        self.board = board
        self.row_length = len(board)
        self.column_length = len(board[0])

    def get_board(self):
        # same implementation as tile puzzle
        return self.board

    def reset(self):
        for row in range(self.row_length): # we loop through every row
            for column in range(self.column_length): # and every column
                self.board[row][column] = False # and set every cell = False, therefore resetting the board to its original state

    def is_legal_move(self, row, col, vertical):
        '''
        VERTICAL --
        If vertical is True:
        The domino covers squares (row, col) and (row + 1, col)

        OTHERWISE

        HORIZONTAL --
        If vertical is False:
        The domino covers squares (row, col) and (row, col + 1).
        '''
        if vertical == True: # if the player is doing vertical moves
            if row + 1 >= self.row_length: # and the move they are trying to make goes out of row bounds
                return False # return False for illegal move
            elif self.board[row][col] == False and self.board[row + 1][col] == False: # if both row cells are empty, this is a legal move
                return True
            else: # otherwise a row cell is occupied and its not
                return False
        else: # if the player is playing horizontal moves
            if col + 1 >= self.column_length: # and the move they are trying to make goes out of column bounds
                return False # return False for illegal move
            elif self.board[row][col] == False and self.board[row][col + 1] == False: # if both column cells are empty, this is a legal move
                return True
            else: # otherwise a column cell is occupied and its not
                return False

    def legal_moves(self, vertical):

        for row in range(self.row_length): # we loop through every row
            for column in range(self.column_length): # and loop through every column
                if self.is_legal_move(row, column, vertical): # if the move is legal using the legal move helper function
                    yield (row, column)  # we yield the legal moves available as a tuple of (row,column)

    def execute_move(self, row, col, vertical):
        if vertical == True: # if the player is playing a vertical move
            # The domino covers squares (row, column) and (row + 1, column)
            self.board[row][col] = True
            self.board[row + 1][col] = True
        else: # if the player is playing a horizontal move
            # The domino covers squares (row, column) and (row, column + 1)
            self.board[row][col] = True
            self.board[row][col + 1] = True

    def game_over(self, vertical):
        # we want to loop through every cell to see if a move is possible
        for row in range(self.row_length): # we loop through every row
            for column in range(self.column_length): # and every column
                if self.is_legal_move(row, column, vertical): # if there is a legal move
                    return False  # there exists at least one legal move so the game isn't over
        return True  # there does not exist at least one legal move so the game is over

    def copy(self):
        # same implementation as tile puzzle
        return DominoesGame(deepcopy(self.board)) 

    def successors(self, vertical):
        for row, column in self.legal_moves(vertical): # unpacking the legal moves into the row and column
            newGame = self.copy() # we make a new game with a copy of the current state so that any changes made to the new game won't affect the original
            newGame.execute_move(row, column, vertical) # apply the move on the new game; This modifies new games board by placing a domino at (row, col) and either (row+1, col) or (row, col+1) depending on what the players move is [veritcal or horizontal]
            yield (row, column), newGame # yield a tuple of ((the move -- row,column), the resulting game state)

    def get_random_move(self, vertical):
        legal_move = list(self.legal_moves(vertical))
        if not legal_move:
            return None  # if the move isnt legal return None
        return random.choice(legal_move) # otherwise return a random legal move


    '''
    This helper function evaluates the board by calculating: how many legal moves the current player has versus how many legal moves the opponent has.
    '''
    def evaluation_helper(self, root_vertical):
        return len(list(self.legal_moves(root_vertical))) - len(list(self.legal_moves(not root_vertical)))
    

    '''
    Alpha-Beta Search with:
    depth: how many layers of moves to simulate
    alpha: best score the maximizing player can perform up to this point
    beta: the best score the minimizing player can perform up to this point
    maximizing: an indicator of whose turn it is (maximizer or minimizer)
    vertical: which player is up at this time
    '''
    def alpha_beta_helper(self, depth, alpha, beta, maximizing, turn_vertical, root_vertical):
        optimal_move = None # we store the optimal move so far
        legal_moves = sorted(list(self.legal_moves(turn_vertical))) # list of legal moves for current player
        nodes_visited = 0 # set the visited nodes = 0

        # base case
        if depth == 0 or self.game_over(turn_vertical): # if we have reached the depth limit or there are no legal moves
            return self.evaluation_helper(root_vertical), None, 1 # we return the evaluation score, None for the move, and the number of leaf nodes visited
        


        if maximizing: # if we are the maximizing player, we want to start with the worst possible value (- infinity), so that any score is better
            optimal_value = float('-inf')

            for row, column in legal_moves: # unpack the list of legal moves into row and column
                new_game = self.copy() # create a copy of the original state so that we don’t affect the original game state when making changes to the new game
                new_game.execute_move(row, column, turn_vertical) # apply the move to the new game by either the vertical or horizontal player

                '''
                recursively evaluate the new resulting game state where:
                depth - 1: we go one level deeper in the tree
                alpha, beta: the current bounds for the search
                False: we switch to the minimizing players turn 
                not vertical: we switch the orientation to the horizontal player orientation / or vertical player orientation if horizontal was just up
                '''
                move_value, child_move, leaf_nodes = new_game.alpha_beta_helper(depth - 1, alpha, beta, False, not turn_vertical, root_vertical) # recursively evaluate the new resulting game state where: depth - 1: we go one level deeper in the tree.alpha, beta: current bounds for pruning.False: it’s now the minimizing player’s turn.not vertical: switches orientation to the opponent
                nodes_visited += leaf_nodes
                if move_value > optimal_value: # if the move value is better then the current / original value
                    optimal_value = move_value # update the optimal value
                    optimal_move = (row, column) # and update the optimal move
                alpha = max(alpha, optimal_value) # we update alpha with the optimal value so far
                if alpha >= beta: # if alpha same or greater than beta the outcome cant improve so we prune bracnhes and break out of loop
                    break
            return optimal_value, optimal_move, nodes_visited # return the best value, best move, and leaf nodes visited for the maximizing player
        
        # otherwise: we are the minimizing player
        else:
            optimal_value = float('inf') # we start with highest possible value
            for row, column in legal_moves: # unpack the list of legal moves into row and column
                new_game = self.copy() # create a copy of the original state so that we don’t affect the original game state when making changes to the new game
                new_game.execute_move(row, column, turn_vertical) # apply the move to the new game by either the vertical or horizontal player

                '''
                recursively evaluate the new resulting game state where:
                depth - 1: we go one level deeper in the tree
                alpha, beta: the current bounds for the search
                True: we switch to maximizing players turn if minimizer was just up
                not vertical: we switch the orientation to the horizontal player orientation / or vertical player orientation if horizontal was just up
                '''

                move_value, child_move, leaf_nodes = new_game.alpha_beta_helper(depth - 1, alpha, beta, True, not turn_vertical, root_vertical)
                nodes_visited += leaf_nodes
                if move_value < optimal_value: # if the move value is better then the current / original value (less then in this case)
                    optimal_value = move_value # update the optimal value
                    optimal_move = (row, column) # and update the optimal move
                beta = min(beta, optimal_value) # we update alpha with the optimal value so far (lower than original)
                if alpha >= beta: # if alpha same or greater than beta the outcome cant improve so we prune bracnhes and break out of loop
                    break
            return optimal_value, optimal_move, nodes_visited # return the best value, best move, and leaf nodes visited for the minimizing player

    # Required
    def get_best_move(self, limit, vertical):

        '''
        we call the recursive alpha-beta search helper function where:
            depth = limit: sets the depth of the search
            alpha=float('-inf'): initial maximizer value
            beta=float('inf'): initial minimizer value
            maximizing=True: the current player is trying to maximize their score, but will switch when the minimizer is trying to minimize their score (according to algorithm max goes first)
            vertical=True: the current players orientation is vertical, but will switch when the other players orientation is horizontal
        '''
        move_value, optimal_move, leaf_nodes_visited = self.alpha_beta_helper(depth=limit, alpha=float('-inf'), beta=float('inf'), maximizing=True, turn_vertical=vertical, root_vertical=vertical) # we unpack the tuple into the move value, the optimal move so far, and the leaf nodes visited

        if optimal_move is None: # if no legal optimal move was found we return None
            return None

        return (optimal_move, move_value, leaf_nodes_visited) # return the move in a tuple of row column
        

