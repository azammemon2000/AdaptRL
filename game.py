import numpy as np
import random
import pickle

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 board stored as a list
        self.current_winner = None  # Track the winner
    
    def reset(self):
        self.board = [' '] * 9
        self.current_winner = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, player):
        if self.board[square] == ' ':
            self.board[square] = player
            if self.check_winner(square, player):
                self.current_winner = player
            return True
        return False

    def check_winner(self, square, player):
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([s == player for s in row]):
            return True

        col_ind = square % 3
        col = [self.board[col_ind+i*3] for i in range(3)]
        if all([s == player for s in col]):
            return True

        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([s == player for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([s == player for s in diagonal2]):
                return True

        return False

    def is_full(self):
        return ' ' not in self.board

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.5, epsilon=0.2):
        self.q_table = {}  # Q-values table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.load_q_table()

        self.moves_table = []
        self.moves_table_user = []

    def get_state(self, board):
        return tuple(board)  # Convert list board into a tuple (hashable)

    def get_q_values(self, state):
        return self.q_table.setdefault(state, np.zeros(9))

    def choose_action(self, board, available_moves, player):
        state = self.get_state(board)
        # print(f"Choose aaction before: {state}")
        state = tuple(' ' if value == player else value for value in state)
        # print(f"Choose aaction after: {state}")
        # if random.uniform(0, 1) < self.epsilon:
        #     return random.choice(available_moves)  # Explore
        q_values = self.get_q_values(state)
        return max(available_moves, key=lambda x: q_values[x])  # Exploit

    # def update_q_table(self, state, action, reward, next_state):
    #     q_values = self.get_q_values(state)
    #     print(f"Q values before update: {q_values}")
    #     next_q_values = self.get_q_values(next_state)
    #     best_next_q = np.max(next_q_values)

        
    #     q_values[action] += self.alpha * (reward + self.gamma * best_next_q - q_values[action])
    #     print(f"Q values after update: {q_values}")
        
    #     # Empty recorded moves for the game
    #     self.moves_table = []
    #     # self.q_table = q_values
    def update_q_table(self, reward, AI_player, Human_player):
        """ Update Q-values for all recorded moves at the end of a game. """
        # AI moves
        for i, (state, action) in enumerate(self.moves_table):
            # print(f"AI State before: {state}, Action: {action}")
            state = tuple(' ' if value == Human_player else value for value in state)
            # print(f"AI State after: {state}, Action: {action}")

            q_values = self.get_q_values(state)
            # print(f"Q values before: {q_values}")
            # print(f"Reward: {self.alpha * reward}")
            # Discount future rewards: Later moves contribute more to learning
            # q_values[action] += self.alpha * (reward + (self.gamma ** (len(self.moves_table) - i)) - q_values[action])
            q_values[action] += self.alpha * reward
            # print(f"Q values after: {q_values}")
            # print("----------------------------------")

        # Update user moves with opposite rewardmvalue
        for i, (state, action) in enumerate(self.moves_table_user):
            # print(f"Human State before: {state}, Action: {action}")
            state = tuple(' ' if value == AI_player else value for value in state)
            # print(f"Human State after: {state}, Action: {action}")
            q_values = self.get_q_values(state)
            # print(f"Q values before: {q_values}")
            # print(f"Reward: {self.alpha * reward}")
            # Discount future rewards: Later moves contribute more to learning
            # q_values[action] += self.alpha * (reward + (self.gamma ** (len(self.moves_table) - i)) - q_values[action])
            q_values[action] += self.alpha * reward * (-1)
            # print(f"Q values after: {q_values}")
            # print("----------------------------------")
    
    def record_move(self, state, action):
        """ Store state-action pairs to update after game result. """
        self.moves_table.append((state, action))  # Store all moves played
        # print(f"Move recorded: {state}, Action: {action}")

    def record_move_user(self, state, action):
        """ Store state-action pairs to update after game result. """
        self.moves_table_user.append((state, action))  # Store all moves played
        # print(f"Move recorded: {state}, Action: {action}")

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
                # print("Memory Loaded!")
                # print(f"Q table loaded: {self.q_table}")
        except FileNotFoundError:
            self.q_table = {}

def play_game(agent, episodes=5000):
    game = TicTacToe()
    rewards = {"win": 10, "loss": -10, "draw": 0}
    
    for episode in range(episodes):
        game.reset()
        state = agent.get_state(game.board)
        player = 'X'  # AI always plays 'X'

        while True:
            available_moves = game.available_moves()
            action = agent.choose_action(game.board, available_moves, player)
            state = agent.get_state(game.board)
            agent.record_move(state, action)
            game.make_move(action, player)
            next_state = agent.get_state(game.board)

            if game.current_winner == player:
                print("AI Winner")
                # agent.update_q_table(state, action, rewards["win"], next_state)
                agent.update_q_table(rewards["win"], player, 'O')
                break
            elif game.is_full():
                print("Full")
                # agent.update_q_table(state, action, rewards["draw"], next_state)
                agent.update_q_table(rewards["draw"], player, 'O')
                break

            # print("User move")
            state = agent.get_state(game.board)
            # opponent_move = random.choice(game.available_moves())  # Opponent plays randomly
            opponent_move = agent.choose_action(game.board, game.available_moves(), 'O')
            # opponent_move = int(input("Your turn? Write from 0 to 8"))  # Opponent plays randomly
            # print(f"opponent move: {opponent_move}")
            game.make_move(opponent_move, 'O')
            agent.record_move_user(state, opponent_move)
            # game.print_board()
            next_state = agent.get_state(game.board)

            if game.current_winner == 'O':
                print("Human Winner")
                # agent.update_q_table(state, action, rewards["loss"], next_state)
                agent.update_q_table(rewards["loss"], player, 'O')
                break
            elif game.is_full():
                print("Full")
                # agent.update_q_table(state, action, rewards["draw"], next_state)
                agent.update_q_table(rewards["draw"], player, 'O')
                break

            # agent.update_q_table(state, action, 0, next_state)
            # agent.update_q_table(0)
            # state = next_state

        if episode % 1000 == 0:
            print(f"Episode {episode}/{episodes} completed.")

    agent.save_q_table()

def human_vs_ai(agent):
    game = TicTacToe()
    game.reset()
    # agent.load_q_table()
    rewards = {"win": 10, "loss": -10, "draw": 0}
    
    human_player = input("Do you want to play as X or O? (X goes first): ").upper()
    while human_player not in ['O', 'X']:
        human_player = input("Do you want to play as X or O? (X goes first): ").upper()

    ai_player = 'O' if human_player == 'X' else 'X'
    
    game.print_board()
    # turn = 'X'
    state = agent.get_state(game.board)
    while True:
        # state = agent.get_state(game.board)
        if game.is_full():
            print("It's a draw!")
            # agent.update_q_table(state, move, rewards["draw"], next_state)
            agent.update_q_table(rewards["draw"], ai_player, human_player)
            break
        
        # print(f"current winner: {game.current_winner}")
        # print(f"full: {game.is_full()}")

        if game.current_winner:
            print(f"Winner is {game.current_winner}!")
            if game.current_winner == human_player:
                print("You won! Congratulations!")
                # agent.update_q_table(state, move, rewards["loss"], next_state)
                agent.update_q_table(rewards["loss"], ai_player, human_player)
            else:
                print("You Loss! Try hard next Time!")
                # agent.update_q_table(state, move, rewards["win"], next_state)
                agent.update_q_table(rewards["win"], ai_player, human_player)
            break

        if human_player == 'X':
            state = agent.get_state(game.board)
            move = int(input("Enter your move (0-8): "))
            while move not in game.available_moves():
                move = int(input("Invalid move! Try again (0-8): "))
            game.make_move(move, human_player)
            agent.record_move_user(state, move)

        else:
            state = agent.get_state(game.board)
            move = agent.choose_action(game.board, game.available_moves(), ai_player)
            print(f"AI chooses {move}")
            game.make_move(move, ai_player)
            agent.record_move(state, move)
            next_state = agent.get_state(game.board)

        game.print_board()

        if game.current_winner or game.is_full():
            if game.current_winner:
                print(f"Winner is {game.current_winner}!")
                if game.current_winner == human_player:
                    print("You won! Congratulations!")
                    # agent.update_q_table(state, move, rewards["loss"], next_state)
                    agent.update_q_table(rewards["loss"], ai_player, human_player)
                else:
                    print("You Loss! Try hard next Time!")
                    # agent.update_q_table(state, move, rewards["win"], next_state)
                    agent.update_q_table(rewards["win"], ai_player, human_player)
            #(265,881 bytes)
            elif game.is_full():
                print("It's a draw!")
                # agent.update_q_table(state, move, rewards["draw"], next_state)
                agent.update_q_table(rewards["draw"], ai_player, human_player)
            break

        if ai_player == 'O':
            state = agent.get_state(game.board)
            # state = next_state
            move = agent.choose_action(game.board, game.available_moves(), ai_player)
            print(f"AI chooses {move}")
            game.make_move(move, ai_player)
            agent.record_move(state, move)
            next_state = agent.get_state(game.board)
        else:
            state = agent.get_state(game.board)
            move = int(input("Enter your move (0-8): "))
            while move not in game.available_moves():
                move = int(input("Invalid move! Try again (0-8): "))
            game.make_move(move, human_player)
            agent.record_move_user(state, move)

        game.print_board()
        state = next_state
    agent.save_q_table()

if __name__ == "__main__":
    agent = QLearningAgent()
    # play_game(agent, episodes=5)  # Train the AI, But due to random ness, it messes up the training
    human_vs_ai(agent)  # Play against AI
