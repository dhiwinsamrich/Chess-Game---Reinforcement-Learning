import chess
import random
import json
import operator
import numpy as np
import pickle
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
import tensorflow as tf
import argparse
import os
import signal
import sys

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.

model = Sequential()
model.add(Dense(20, input_shape=(65,) , kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))    # Same number of outputs as possible actions
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
np.set_printoptions(threshold=np.inf)
state_board=np.zeros((1,65)) # The array representing the board of 8*8 and also a selected move from the possible ones
# Value of every piece
switch = {
    'p': 10, 'P': -10, 'q': 90, 'Q': -90, 'n': 30, 'N': -30, 
    'r': 50, 'R': -50, 'b': 30, 'B': -30, 'k': 900, 'K': -900, 
    'None': 0
}

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_games', type=float, default=100)
parser.add_argument('--winner_reward', type=float, default=1)
parser.add_argument('--loser_malus', type=float, default=-1)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--decremental_epsilon', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.05)
args, unknown = parser.parse_known_args(sys.argv)

arguments = {'training_games': args.number_of_games, 'winner_reward': args.winner_reward, 'loser_malus': args.loser_malus, 
             'epsilon': args.epsilon, 'decremental_epsilon': args.decremental_epsilon, 'gamma': args.gamma}
general_moves = {}

steps = 1000
training_games = int(arguments['training_games']) if (arguments['training_games'] is not None) else 100
winner_reward = int(arguments['winner_reward']) if (arguments['winner_reward'] is not None) else 1
loser_malus = int(arguments['loser_malus']) if (arguments['loser_malus'] is not None) else -1
epsilon = float(arguments['epsilon']) if (arguments['epsilon'] is not None) else 1  # Probability of doing a random move
decremental_epsilon = float(arguments['decremental_epsilon']) if (arguments['decremental_epsilon'] is not None) else 1/training_games  # Each game we play we want to decrease the probability of random move
gamma = float(arguments['gamma']) if (arguments['gamma'] is not None) else 0.05  # Discounted future reward. How much we care about steps further in time

print("Training the Deep-Q-Network with parameters: ")
print(f"Number of training games: {training_games}")
print(f"Winner Reward: {winner_reward}")
print(f"Loser Malus: {loser_malus}")
print(f"Epsilon: {epsilon}")
print(f"Decremental Epsilon: {decremental_epsilon}")
print(f"Gamma: {gamma}")

def evaluate_board(turn):  # Evaluate the board following the value of each piece
    l = chess.SQUARES
    total = 0
    mult = 1 if turn else -1
    a = 0
    for i in l:
        total += (mult * switch[str(board.piece_at(i))])
        state_board[0][a] = mult * switch[str(board.piece_at(i))]  # Update the state_board variable used for predictions
        a += 1
    return total

def get_int(move):  # Give the int representation(maping) of the move from the dictionary to give it as input for the deep neural network
    try:
        return general_moves[str(move)]
    except:
        general_moves[str(move)] = len(general_moves)
        return general_moves[str(move)]

def reward(fen_history, moves, lose_fen, lose_moves):
    maxi = len(fen_history)
    for i in range(maxi):
        gamma = 1 / len(fen_history)
        fen_history[i][0][64] = get_int(moves[i])
        # Adjust the reward depending on how good the position is after each move
        model.train_on_batch(np.array(fen_history[i]), model.predict(np.array(fen_history[i])) + winner_reward * (gamma * i))
    maxi = len(lose_fen)
    for i in range(maxi):
        gamma = 1 / len(lose_fen)
        lose_fen[i][0][64] = get_int(lose_moves[i])
        model.train_on_batch(np.array(lose_fen[i]), model.predict(np.array(lose_fen[i])) + loser_malus * (gamma * i))
        
winners = {}  # Variable for counting number of wins of each player

def save_model():
  try:
    with open('generalized_moves.json', 'w') as fp:  # Save the mapping Move/Index to be used on development
        json.dump(general_moves, fp)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # graph = tf.get_default_graph()
    # with graph.as_default():
    model.save("model.h5")
    print("Model saved.")
    print("\n\n")
    print("Model and Generalized Moves saved successfully.")
  except Exception as e:
    print(f"An error occurred while saving the model:{e}")

# Keyboard interrupt handler
def signal_handler(sig, frame):
    print("Saving model due to keyboard interrupt...")
    save_model()
    sys.exit(0)

# Set the signal handler for keyboard interrupt
signal.signal(signal.SIGINT, signal_handler)

for joum in range(steps):
    i = 0
    evaluation_history = []
    all_number_of_moves = []
    board = chess.Board()
    epsilon = 1
    decremental_epsilon = 1 / training_games
    while i < training_games:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("/------------------ Training -----------------/")
        print(f"Step ({joum}/{steps})")
        print(f"Game N°{i}")
        print(f"WINNERS COUNT: \n{winners}")
        print(f"Number of remaining training games: {training_games - i}")
        print(f"Winner Reward: {winner_reward}")
        print(f"Loser Malus: {loser_malus}")
        print(f"Epsilon: {epsilon}")
        print(f"Decremental Epsilon: {decremental_epsilon}")
        print(f"Gamma: {gamma}")
        print(f"Rewards for White: {winner_reward}, Black: {loser_malus}")
        
        fen_history = []
        black_moves = []
        white_moves = []
        black_fen_history = []
        white_fen_history = []
        all_states = []
        all_moves = []
        number_of_moves = 0
        evaluation_history = []
        
        while not board.is_game_over():
            number_of_moves += 1
            if np.random.rand() <= epsilon:
                nmov = random.randint(0, board.legal_moves.count())
                cnt = 0
                for k in board.legal_moves:
                    if cnt == nmov:
                        god_damn_move = str(k)
                    cnt += 1
            else:
                evaluate_board(True)
                Q = {}
                for kr in board.legal_moves:
                    br = get_int(kr)
                    state_board[0][64] = br
                    Q[kr] = model.predict(state_board)  # Q-values predictions for every action possible with the actual state
                god_damn_move = max(Q.items(), key=operator.itemgetter(1))[0]  # Get the move with the highest Q-value

            base_evaluation = evaluate_board(board.turn)
            fen = str(board.fen())
            all_states.append(np.array(state_board, copy=True))
            all_moves.append(np.array(god_damn_move, copy=True))
            evaluation_history.append(base_evaluation)
            if board.turn:
                white_moves.append(god_damn_move)
                white_fen_history.append(np.array(state_board, copy=True))
            else:
                black_moves.append(god_damn_move)
                black_fen_history.append(np.array(state_board, copy=True))

            if chess.Move.from_uci(str(god_damn_move)) in board.legal_moves:
                board.push(chess.Move.from_uci(str(god_damn_move)))
            else:
                print("illegal move! Invalidation!")
                break
        if board.result() == "1-0":
            reward(fen_history, white_moves, black_fen_history, black_moves)  # White wins
            winners["White"] = winners.get("White", 0) + 1
        elif board.result() == "0-1":
            reward(fen_history, black_moves, white_fen_history, white_moves)  # Black wins
            winners["Black"] = winners.get("Black", 0) + 1
        else:
            print("DRAW!!")

        epsilon -= decremental_epsilon
        i += 1
        
    if joum % 100 == 0:
        save_model()

    save_model()