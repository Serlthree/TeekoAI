import random
import copy
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
    
    def successors(self, state, player = False):
        """ Generates all possible successor states for the current state of the game.
        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method. Any
                modifications should be done on a deep copy of the state.

        Returns:
            list of states: a list of all possible successor states for the current state.
        """
        # check drop phase (less than 8 pieces on the board)
        drop_phase = True
        cnt = 0
        for row in state:
            cnt += row.count('b') + row.count('r')
        if cnt == 8:
            drop_phase = False
        #if drop phase generate all possible states
        #we play with self.my_piece
        #TODO: tests for successor states
        piece = self.my_piece
        if player == True:
            piece = self.opp
        states = []
        if drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        #copy the state
                        new_state = copy.deepcopy(state)
                        new_state[i][j] = piece
                        states.append(new_state)
        else:
            #generate all possible moves
            left = [-1, 0, 1, 0]
            right = [0, 1, 0, -1]
            for i in range(5):
                for j in range(5):
                    if state[i][j] == piece:
                        #move the piece to an adjacent empty cell
                        #NOTE: duplicates don't occur because moving vacates the source and occupies the destination
                        for k in range(4):
                            x = i + left[k]
                            y = j + right[k]
                            if x >= 0 and x < 5 and y >= 0 and y < 5 and state[x][y] == ' ':
                                #copy the state
                                new_state = copy.deepcopy(state)
                                new_state[i][j] = ' '
                                new_state[x][y] = piece
                                states.append(new_state)
        return states
    
    def max_value(self, state, depth, alpha, beta):
        """ Implements max-value function for the minimax algorithm. This function
        will select the move with the highest heuristic value at the end of the search.
        NOTE: Must select within 5 seconds or less.
        
        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method. Any
                modifications should be done on a deep copy of the state.
            depth (int): the current depth in the search tree
            alpha (int): the alpha value for the alpha-beta pruning algorithm 
                maximum heuristic value that the AI can force the opponent to get
            beta (int): the beta value for the alpha-beta pruning algorithm
                minimum heuristic value that the opponent can force the AI to get

        Returns:
            int: the heuristic value of the best move for the AI player
        """
        #if the game is over or the depth is 0 return the heuristic value
        if self.game_value(state) != 0 or depth == 0:
            return self.heuristic_game_value(state)
        #generate all possible states
        for s in self.successors(state):
            v = self.min_value(s, depth-1, alpha, beta)
            if v > alpha:
                alpha = v
            if alpha >= beta:
                return beta
        return alpha

    def min_value(self, state, depth, alpha, beta):
        """ Implements min-value function for the minimax algorithm. This function
        will select the move with the lowest heuristic value at the end of the search.
        NOTE: Must select within 5 seconds or less.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method. Any
                modifications should be done on a deep copy of the state.
            depth (int): the current depth in the search tree
            alpha (int): the alpha value for the alpha-beta pruning algorithm 
                maximum heuristic value that the AI can force the opponent to get
            beta (int): the beta value for the alpha-beta pruning algorithm
                minimum heuristic value that the opponent can force the AI to get

        Returns:
            int: the heuristic value of the best move for the opponent player
        """
        #if the game is over or the depth is 0 return the heuristic value
        if self.game_value(state) != 0 or depth == 0:
            return self.heuristic_game_value(state)
        #generate all possible states
        for s in self.successors(state, True):
            v = self.max_value(s, depth-1, alpha, beta)
            if v < beta:
                beta = v
            if alpha >= beta:
                return alpha
            #if min == -1:
            #    print("State: ", s)
        return beta
    
    def get_move_from_two_states(self, state1, state2, drop_phase):
        """ Returns the move that was made between two states. This function is useful 
        for converting states for the minimax into moves for the AI to make.
        
        Args:
            state1 (list of lists): the first state
            state2 (list of lists): the second state
            drop_phase (bool): True if the game is in the drop phase, False otherwise

        Returns:
            list: a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        if drop_phase:
            for i in range(5):
                for j in range(5):
                    if state1[i][j] != state2[i][j]:
                        return [(i, j)]
        else:
            a, b, c, d = 0, 0, 0, 0
            for i in range(5):
                for j in range(5):
                    if state1[i][j] != state2[i][j]:
                        if state1[i][j] == ' ':
                            a, b = i, j
                        else:
                            c, d = i, j
            return [(a, b), (c, d)]
        return None

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = True   # TODO: detect drop phase
        #detect drop phase
        cnt = 0
        for row in state:
            cnt += row.count('b') + row.count('r')
        if cnt == 8:
            drop_phase = False
        #check time per move
        #time1 = time.time()
        OPENING_DEPTH = 2
        DEPTH = 5
        #generate all possible states
        self.successors(state)
        #run the minimax algorithm
        move = None
        max = -1.1
        for s in self.successors(state):
            #plays autowins if they can
            if self.game_value(s) == 1:
                return self.get_move_from_two_states(state, s, drop_phase)
            if drop_phase:
                v = self.min_value(s, OPENING_DEPTH, -1.1, 1.1)
            else:
                v = self.min_value(s, DEPTH, -1.1, 1.1)
            if v > max:
                max = v
                move = self.get_move_from_two_states(state, s, drop_phase)
        #time2 = time.time()
        #print("Time taken: ", time2-time1, " seconds")
        #print("Evaluation: ", max)
        # ensure the destination (row,col) tuple is at the beginning of the move list
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1
        # TODO: tests for diagonal and box wins
        # TODO: check \ diagonal wins
        #because of the 5x5 board, we only need to check the 2x2 subgrid
        for i in range(2):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    return 1 if state[i][j]==self.my_piece else -1
        # TODO: check / diagonal wins
        for i in range(2):
            for j in range(2):
                #0,3 1,2 2,1 3,0
                if state[i][j+3] != ' ' and state[i][j+3] == state[i+1][j+2] == state[i+2][j+1] == state[i+3][j]:
                    return 1 if state[i][j+3]==self.my_piece else -1
        # TODO: check box wins
        #only need to check the 4x4 subgrid
        for i in range(4):
            for j in range(4):
                #0,0 0,1 1,0 1,1
                if state[i][j] != ' ' and state[i][j] == state[i][j+1] == state[i+1][j] == state[i+1][j+1]:
                    return 1 if state[i][j]==self.my_piece else -1

        return 0 # no winner yet
    
    def heuristic_game_value(self, state):
        """  Generates a heuristic value for the current state of the game. This value
        is used to determine the best move for the AI to make.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method. Any
                modifications should be done on a deep copy of the state.
        
        Returns:
            float: the heuristic value of the current state
        """
        if self.game_value(state) != 0:
            return self.game_value(state)
    
        state_list = []
        for i in range(5):
            for j in range(5):
                if state[i][j] == 'r':
                    state_list.append(1.0)
                elif state[i][j] == 'b':
                    state_list.append(-1.0)
                else:
                    state_list.append(0.0)
        #manually set the weights (from linear regression model)
        A = [ 0.1373,  0.0445, -0.0356,  0.0032,  0.0709,  0.0112, -0.0409, -0.0611,
         -0.0177,  0.1151, -0.0509, -0.0795, -0.1177, -0.0405,  0.0081,  0.0071,
         -0.0458, -0.0918, -0.0757, -0.0410,  0.0276, -0.0143, -0.0136, -0.0202,
          0.0443]
        B = [0.2216]
        #calculate the value
        value = 0
        for i in range(25):
            value += A[i]*state_list[i]
        value += B[0]
        if self.my_piece == 'r':
            value = -value
        #add epsilon to make sure it doesn't get stuck in a loop
        epsilon = random.uniform(-0.125, 0.125)
        value += epsilon
        #need to bound the value between -1 and 1
        if value > 1:
            value = 1
        elif value < -1:
            value = -1
        return value
        

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")

if __name__ == "__main__":
    main()
