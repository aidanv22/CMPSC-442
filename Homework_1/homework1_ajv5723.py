############################################################
# CMPSC/DS 442: Uninformed Search
############################################################

student_name = "Aidan Vesci"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
from math import comb
import random
from copy import deepcopy
from queue import Queue

############################################################
# Section 1: N-Queens
############################################################

def num_placements_entire(n):
    # as long as a board exists and isnt negative
    if n >= 0:
        # since there are n^2 total squares, and we ignore chess constraints, we can do a choose function of n^2 choose n
        return comb(n*n, n)
    # this is an edge case for a negative output
    else:
        return 0

def num_placements_one_in_row(n):
    # as long as a board exists and isnt negative
    if n >= 0:
        # since for each of the n rows, there are n column choices, and we enforce one queen per row, we can pick one column per row, which is n choices repeated n times
        return (n**n)
    # this is an edge case for a negative output
    else:
        return 0

def n_queens_valid(board):
    # comparing each pair of placed queens, the board is a list where the index is the row and the value is the column where that row has its queen placed
    # board[i] = column at row i
    for i in range(len(board)): # check every queen
        for j in range(i+1, len(board)): # with every next queen
            # if the queens are in the same column, then False
            if board[i] == board[j]:
                return False
            # check diagonal, two queens are on a diagonal if and only if row diff = col diff
            if abs(board[i] - board[j]) == abs(i-j):
                return False
    return True

def n_queens_helper(n, board):
    # board is a list where board[i] = column of queen in row i
    row = len(board) # which row we are about to fill
    # The next row to place a queen in is exactly the length of the partial list
    '''
    Base case: If weve placed queens in all n rows, the partial placement is a complete solution 
    board[:] returns a copy so later backtracking can't mutate what we yielded
    yield emits this solution and then the return stops this call.
    '''
    if row == n: # base case: all rows filled
        yield board[:]  # yield a full valid solution
        return
    

    # try placing a queen in each column of this row
    for col in range(n):
        '''
        Explore every column for the current row (the branching step)
        Lets say we adding a queen at (row, col)
        Using board + [col] creates a new list without in-place mutation so we can backtrack
        '''

        queen_candidate = board + [col]        # put queen at (row, col)

        # only continue down branches where the new queen doesn’t conflict with earlier ones
        # n_queens_valid should ensure no same column and no diagonal conflicts
       
        if n_queens_valid(queen_candidate):    # only continue if placement is safe
            # recurse to fill next row with updated board
            # since we recurse before trying next column it is a DFS
            yield from n_queens_helper(n, queen_candidate)

def n_queens_solutions(n):
    if n < 0:
        raise ValueError("n must be >= 0")
    else:
        yield from n_queens_helper(n, [])    # start search with empty board
    

############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.row_length = len(board)
        self.column_length = len(board[0])

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        # center
        self.board[row][col] = not self.board[row][col] # toggle the selected cell -- the "not": Flips T<->F

        # left
        if col - 1 >= 0: # prevents negative indexing
            self.board[row][col - 1] = not self.board[row][col - 1] # if there exists a cell to the left toggle the cell to the left (same row, next column)

        # right
        if col + 1 < self.column_length:  # preventing index error
            self.board[row][col + 1] = not self.board[row][col + 1] # if there exists a cell to the right, toggle the cell to the right (same row, next column)

        # up
        if row - 1 >= 0: # preventing negative indexing
            self.board[row - 1][col] = not self.board[row - 1][col] # if there exists a cell above the selected cell, toggle the cell above it (same column, above row)


        # down
        if row + 1 < self.row_length: # preventing index error
            self.board[row + 1][col] = not self.board[row + 1][col] # if there exists a cell below the selected cell, toggle the cell below it (same column, below row)


    def scramble(self): # we start off with an "all-off" or FALSE board
        # together these two loops will visit every cell (row,column) once
        for row in range(self.row_length): # loop through every row
            for column in range(self.column_length): # loop through every column
                if random.random() < 0.5: # 50/50 chance of a cell being true or false
                    self.perform_move(row, column)

    def is_solved(self):
         # together these two loops will visit every cell (row,column) once
        for row in range(self.row_length): # loop through every row
            for column in range(self.column_length): # loop through every column
                if self.board[row][column]:      # a light is on / TRUE
                    return False
        return True

    def copy(self):
        return LightsOutPuzzle(deepcopy(self.board))

    def successors(self):
         # together these two loops will visit every cell (row,column) once
        for row in range(self.row_length): # loop through every row
            for column in range(self.column_length): # loop through every column
                new_puzzle = self.copy() # create a new board thats a copy of original
                '''
                Apply the Lights Out move at (row, column) to the copy: toggle the pressed cell and its neighbors
                new_puzzle now represents the resulting state after that move.
                '''
                new_puzzle.perform_move(row,column)
                yield ((row, column),new_puzzle) # Yield a successor: the move (row, column) -- a tuple, and the new puzzle state after applying that move


    def find_solution(self):
        explored = set() # intialize set of board states that have already been visited
        frontier = [([], self.copy())] # create a BFS queue with the list of moves and puzzle as a tuple
        start_state = tuple(tuple(row) for row in self.board) # converts board "list of lists" to tuple of tuples
        explored.add(start_state) # add the start state to the list of states already explored

        if self.is_solved(): # if board solve return empty list with no moves
            return []

        while len(frontier) > 0: # as long as frontier not empty
            moves_taken, current_puzzle = frontier.pop(0) # since we are using BFS and FIFO, we remove the first element and unpack into moves taken and the current puzzle state
            '''
            below we iterate over all possible moves from current puzzle state
            Each of the succesors is  the move performed in (row,column) form, as well as the new resulting puzzle
            '''
            for move, new_puzzle in current_puzzle.successors(): 
                new_state = tuple(tuple(row) for row in new_puzzle.board) # we convert that succesors board into a tuple of tuple format similar to how we did above
                if new_state not in explored: # if this board configuration hasn't been explored before go one, else skip it
                    if new_puzzle.is_solved(): # if this succesor has already been solved:
                        return moves_taken + [move] # return the moves taken and the final move
                    explored.add(new_state) # add the new board configuration to explored so we can avoid visiting it again
                    frontier.append((moves_taken + [move], new_puzzle)) # add this new board state to the BFS queue with a tuple of the updated path: moves taken and final move, as well as the new board configuration

        return None # if puzzle is unsolvable
        

def make_puzzle(rows, cols):
    # create a 2D list that is FALSE for all columns and then FALSE for all rows
    make_board = [[False for _ in range(cols)] for _ in range(rows)]
    return LightsOutPuzzle(make_board)

## CHECK THE GUI FOR THIS WITH THE TA

############################################################
# Section 3: Linear Disk Movement
############################################################

def solve_identical_disks(length, n):

    # if there are no disks to move return an empty list
    if n == 0:
        return []

    # Initial and goal states
    start_state = tuple([1]*n + [0]*(length - n)) # the start state is in tuple form of (n disks, and 0s (length-n))
    goal_state = tuple([0]*(length - n) + [1]*n) # the goal state is flipped in tuple form of (0s (length-n), n disks)


    # BFS queue with FIFO attributes
    BFS_Queue = Queue()

    # we add the starting node as well as moves so far to the queue in tuple form
    BFS_Queue.put((start_state, []))
    explored = {start_state} # keep a set of visited states to avoid going through the same scenario more than once

    while not BFS_Queue.empty(): # while the BFS queue still has states to explore
        state, moves_taken = BFS_Queue.get() # finds the next state and moves_taken to get it

        # check if the current state is the goal state, if it is return the list of moves used to get there
        if state == goal_state:
            return moves_taken

        '''
        this loop iterates over positions of i where a disk could either move: 
        one spot to the right (i+1) 
        or 
        two spots to the right (i+2)
        '''

        for i in range(length - 1): # since it only moves forward we need to avoid out of bounds
            # Single jump of i: i+1
            if state[i] == 1 and state[i+1] == 0: # sees if a one step move is legal with a disk at i and an empty space at i+1
                new_state = list(state) # this tuple gets copied to a list for later mutation
                new_state[i], new_state[i+1] = 0, 1 # applies the move  from i to i+1
                new_state = tuple(new_state) # converting state back to a tuple

                if new_state not in explored: # avoiding going back over states we've already seen
                    explored.add(new_state) # if we havent seen it add it to the set
                    BFS_Queue.put((new_state, moves_taken + [(i, i+1)])) # we then Enqueue the succesor disk and update the path to get here

            # Double jump of i: i+2 (could also jump over one disk)
            if i + 2 < length and state[i] == 1 and state[i+1] == 1 and state[i+2] == 0: # sees if a two step move is legal with a disk at i+1 that the disk at i will jump over and an empty space at i+2 
                new_state = list(state) # this tuple gets copied to a list for later mutation as shown above
                new_state[i], new_state[i+2] = 0, 1  # applies the move  from i to i+2
                new_state = tuple(new_state) # converting state back to a tuple as shown above

                if new_state not in explored: # again avoiding going back over states we've already seen
                    explored.add(new_state) # again if we havent seen it add it to the set
                    BFS_Queue.put((new_state, moves_taken + [(i, i+2)])) # again we then Enqueue the succesor disk and update the path to get here

    # if there are no possible moves return None
    return None

def solve_distinct_disks(length, n):
     # if there are no disks to move return an empty list
    if n == 0:
        return []

    # Initial and goal states
    start_state = tuple(list(range(1, n + 1)) + [0] * (length - n)) # leftmost n cells are filled with disks, the rest are filled with zeros
    goal_state = tuple([0] * (length - n) + sorted(range(1, n + 1), reverse=True)) # in the goal state the rightmost n cells are filled with disks in reverse order, and the rest of the left is filled with zeros


    # BFS queue with FIFO attributes
    BFS_Queue = Queue()

    # we add the starting node as well as moves so far to the queue in tuple form
    BFS_Queue.put((start_state, []))
    explored = {start_state} # keep a set of visited states to avoid going through the same scenario more than once

    while not BFS_Queue.empty(): # while the BFS queue still has states to explore
        state, moves_taken = BFS_Queue.get() # finds the next state and moves_taken to get it

        # check if the current state is the goal state, if it is return the list of moves used to get there
        if state == goal_state:
            return moves_taken

        '''
        this loop iterates over positions of i where a disk could either move: 
        one spot to the right (i+1) 
        or 
        two spots to the right (i+2)
        '''

        for i in range(length): # since we can move backwards we need to go to the full length to potenitally check every index

            # RIGHT

            # Single jump of i right: i+1
            if i+1 < length and state[i] != 0 and state[i+1] == 0: # sees if a one step move is legal (in bounds) with a disk at i and an empty space at i+1 (we use state[i] != 0 because of distinct disks)
                new_state = list(state) # this tuple gets copied to a list for later mutation
                new_state[i], new_state[i+1] = 0, state[i] # applies the move one step to the right (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple

                if new_state not in explored: # avoiding going back over states we've already seen
                    explored.add(new_state) # if we havent seen it add it to the set
                    BFS_Queue.put((new_state, moves_taken + [(i, i+1)])) # we then Enqueue the succesor disk and update the path to get here

            # Double jump of i right: i+2 (could also jump over one disk)
            if i + 2 < length and state[i] != 0 and state[i+1] != 0 and state[i+2] == 0: # sees if a two step move is legal (index not out of bounds), with a disk at i+1 that the disk at i will jump over and an empty space at i+2 (we use state[i] != 0, state[i+1] != 0  because of distinct disks)
                new_state = list(state) # this tuple gets copied to a list for later mutation as shown above
                new_state[i], new_state[i+2] = 0, state[i]  # applies the move  from i to i+2 (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple as shown above

                if new_state not in explored: # again avoiding going back over states we've already seen
                    explored.add(new_state) # again if we havent seen it add it to the set
                    BFS_Queue.put((new_state, moves_taken + [(i, i+2)])) # again we then Enqueue the succesor disk and update the path to get here
            

            # LEFT

            # Single jump of i left: i+1
            if i-1 >= 0 and state[i] != 0 and state[i-1] == 0: # sees if a one step move is legal (index not out of bounds) with a disk at i and an empty space at i-1 (we use state[i] != 0 because of distinct disks)
                new_state = list(state) # this tuple gets copied to a list for later mutation
                new_state[i], new_state[i-1] = 0, state[i] # applies the move  from i to i-1 (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple

                if new_state not in explored: # avoiding going back over states we've already seen
                    explored.add(new_state) # if we havent seen it add it to the set
                    BFS_Queue.put((new_state, moves_taken + [(i, i-1)])) # we then Enqueue the succesor disk and update the path to get here

            # Double jump of i left: i+2 (could also jump over one disk)
            if i - 2 >= 0 and state[i] != 0 and state[i-1] != 0 and state[i-2] == 0: # sees if a two step move is legal (index not out of bounds), with a disk at i-1 that the disk at i will jump over and an empty space at i-2 (we use state[i] != 0, state[i-1] != 0 because of distinct disks)
                new_state = list(state) # this tuple gets copied to a list for later mutation as shown above
                new_state[i], new_state[i-2] = 0, state[i]  # applies the move  from i to i-2 (we need to keep track of each distinct disk label in this problem)
                new_state = tuple(new_state) # converting state back to a tuple as shown above

                if new_state not in explored: # again avoiding going back over states we've already seen
                    explored.add(new_state) # again if we havent seen it add it to the set
                    BFS_Queue.put((new_state, moves_taken + [(i, i-2)])) # again we then Enqueue the succesor disk and update the path to get here
    
    # if there are no possible moves return None
    return None

