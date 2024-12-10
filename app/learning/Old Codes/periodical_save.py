import chess
import random
import json
import operator
import numpy as np
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
import argparse
import os
import signal
import sys

# Create the neural network model
model = Sequential()
model.add(Dense(20, input_shape=(65,), kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
np.set_printoptions(threshold=np.inf)

state_board = np.zeros((1, 65))  # Array representing the board (8x8) and a selected move
switch = {
    'p': 10, 'P': -10, 'q': 90, 'Q': -90, 'n': 30, 'N': -30,
    'r': 50, 'R': -50, 'b': 30, 'B': -30, 'k': 900, 'K': -900,
    'None': 0
}

# Parse training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--number_of_games', type=float, default=100)
parser.add_argument('--winner_reward', type=float, default=1)
parser.add_argument('--loser_malus', type=float, default=-1)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--decremental_epsilon', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.05)
args, unknown = parser.parse_known_args(sys.argv)

arguments = {
    'training_games': args.number_of_games, 'winner_reward': args.winner_reward,
    'loser_malus': args.loser_malus, 'epsilon': args.epsilon,
    'decremental_epsilon': args.decremental_epsilon, 'gamma': args.gamma
}
general_moves = {}
steps = 1000
training_games = int(arguments['training_games'])
winner_reward = int(arguments['winner_reward'])
loser_malus = int(arguments['loser_malus'])
epsilon = float(arguments['epsilon'])
decremental_epsilon = float(arguments['decremental_epsilon'])
gamma = float(arguments['gamma'])

print("Training the Deep-Q-Network with parameters:")
print(f"Number of training games: {training_games}")
print(f"Winner Reward: {winner_reward}")
print(f"Loser Malus: {loser_malus}")
print(f"Epsilon: {epsilon}")
print(f"Decremental Epsilon: {decremental_epsilon}")
print(f"Gamma: {gamma}")
print(f"Rewards for White: {winner_reward}, Black: {loser_malus}")


winners = {}  # Dictionary to track the number of wins for each player

def evaluate_board(turn):
    total = 0
    mult = 1 if turn else -1
    for idx, square in enumerate(chess.SQUARES):
        piece = str(board.piece_at(square))
        state_board[0][idx] = mult * switch.get(piece, 0)
        total += state_board[0][idx]
    return total

def get_int(move):
    try:
        return general_moves[str(move)]
    except KeyError:
        general_moves[str(move)] = len(general_moves)
        return general_moves[str(move)]

def reward(fen_history, moves, lose_fen, lose_moves):
    # Reward winning moves
    for i, state in enumerate(fen_history):
        if i < len(moves):  # Ensure `moves[i]` exists
            gamma_factor = 1 / len(fen_history)
            state[0][64] = get_int(moves[i])
            model.train_on_batch(
                np.array(state),
                model.predict(np.array(state)) + winner_reward * (gamma_factor * i)
            )
        else:
            print(f"Warning: Attempted to access moves[{i}] but only {len(moves)} moves available.")
    
    # Penalize losing moves
    for i, state in enumerate(lose_fen):
        if i < len(lose_moves):  # Ensure `lose_moves[i]` exists
            gamma_factor = 1 / len(lose_fen)
            state[0][64] = get_int(lose_moves[i])
            model.train_on_batch(
                np.array(state),
                model.predict(np.array(state)) + loser_malus * (gamma_factor * i)
            )
        else:
            print(f"Warning: Attempted to access lose_moves[{i}] but only {len(lose_moves)} moves available.")


def save_model():
    try:
        with open('generalized_moves.json', 'w') as fp:
            json.dump(general_moves, fp)

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("model.weights.h5")
        print("Model and generalized moves saved successfully!")

    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

def signal_handler(sig, frame):
    print("Saving model due to keyboard interrupt...")
    save_model()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

for joum in range(steps):
    i = 0
    board = chess.Board()
    while i < training_games:
        fen_history = []
        black_moves = []
        white_moves = []
        black_fen_history = []
        white_fen_history = []

        while not board.is_game_over():
            if np.random.rand() <= epsilon:
                # Select a random legal move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    god_damn_move = str(random.choice(legal_moves))
                else:
                    print("No legal moves available.")
                    break
            else:
                evaluate_board(board.turn)
                Q = {}
                for legal_move in board.legal_moves:
                    move_idx = get_int(legal_move)
                    state_board[0][64] = move_idx
                    Q[legal_move] = model.predict(state_board)
                
                if Q:
                    god_damn_move = max(Q.items(), key=operator.itemgetter(1))[0]
                else:
                    print("No valid moves predicted.")
                    break

            # Push the move if it's valid
            try:
                if isinstance(god_damn_move, chess.Move):  
                    move = god_damn_move
                else:
                    move = chess.Move.from_uci(str(god_damn_move))
                    
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print(f"Illegal move attempted: {god_damn_move}")
                    break
            except ValueError:
                print(f"Error parsing move: {god_damn_move}")
                break

            fen_history.append(np.array(state_board, copy=True))

            if board.turn:
                white_moves.append(god_damn_move)
                white_fen_history.append(np.array(state_board, copy=True))
            else:
                black_moves.append(god_damn_move)
                black_fen_history.append(np.array(state_board, copy=True))
                
        if len(fen_history) != len(white_moves):
            print(f"Mismatch: fen_history({len(fen_history)}) != white_moves({len(white_moves)})")
        if len(black_fen_history) != len(black_moves):
            print(f"Mismatch: black_fen_history({len(black_fen_history)}) != black_moves({len(black_moves)})")

        # Handle game result
        if board.result() == "1-0":
            reward(fen_history, white_moves, black_fen_history, black_moves)
            winners["White"] = winners.get("White", 0) + 1
        elif board.result() == "0-1":
            reward(fen_history, black_moves, white_fen_history, white_moves)
            winners["Black"] = winners.get("Black", 0) + 1
        else:
            print("Game ended in a draw.")

        epsilon = max(0.01, epsilon - decremental_epsilon)  # Decay epsilon with a floor value
        i += 1

    # Save the model every 100 steps
    if joum % 100 == 0:
        save_model()

save_model()  # Final save after training
