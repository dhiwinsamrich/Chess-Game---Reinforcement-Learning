import chess
import random
import json
import operator
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import argparse
import os
import multiprocessing as mp  # For multiprocessing
from functools import partial  # To pass additional arguments in multiprocessing

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
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
state_board = np.zeros((1, 65))  # The array representing the board and also a selected move

# Value of every piece
switch = {
    'p': 10, 'P': -10, 'q': 90, 'Q': -90,
    'n': 30, 'N': -30, 'r': 50, 'R': -50,
    'b': 30, 'B': -30, 'k': 900, 'K': -900, 'None': 0
}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--number_of_games', type=int, default=100)
parser.add_argument('--winner_reward', type=float, default=1)
parser.add_argument('--loser_malus', type=float, default=-1)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--decremental_epsilon', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.05)
args, unknown = parser.parse_known_args()

training_games = args.number_of_games
winner_reward = args.winner_reward
loser_malus = args.loser_malus
epsilon = args.epsilon
decremental_epsilon = args.decremental_epsilon
gamma = args.gamma

general_moves = {}
steps = 1000
winners = {}  # Variable for counting the number of wins of each player


def evaluate_board(turn, board):
    """Evaluate the board following the value of each piece."""
    l = chess.SQUARES
    total = 0
    mult = 1 if turn else -1
    a = 0
    for i in l:
        total += mult * switch[str(board.piece_at(i))]
        state_board[0][a] = mult * switch[str(board.piece_at(i))]
        a += 1
    return total


def get_int(move):
    """Map the move to an integer representation."""
    if move not in general_moves:
        general_moves[move] = len(general_moves)
    return general_moves[move]


def reward(fen_history, moves, lose_fen, lose_moves):
    """Assign final rewards at the end of the game."""
    for i in range(len(fen_history)):
        gamma_step = 1 / len(fen_history)
        fen_history[i][0][64] = get_int(moves[i])
        model.train_on_batch(
            np.array(fen_history[i]),
            model.predict(np.array(fen_history[i])) + winner_reward * (gamma_step * i),
        )

    for i in range(len(lose_fen)):
        gamma_step = 1 / len(lose_fen)
        lose_fen[i][0][64] = get_int(lose_moves[i])
        model.train_on_batch(
            np.array(lose_fen[i]),
            model.predict(np.array(lose_fen[i])) + loser_malus * (gamma_step * i),
        )


def simulate_game(_):
    """Simulate a single game."""
    global epsilon
    board = chess.Board()
    fen_history = []
    black_moves = []
    white_moves = []
    black_fen_history = []
    white_fen_history = []
    number_of_moves = 0

    while not board.is_game_over():
        number_of_moves += 1
        if np.random.rand() <= epsilon:
            # Random move
            nmov = random.randint(0, board.legal_moves.count() - 1)
            move = list(board.legal_moves)[nmov]
        else:
            # Q-move based on predictions
            evaluate_board(board.turn, board)
            Q = {}
            for kr in board.legal_moves:
                br = get_int(kr)
                state_board[0][64] = br
                Q[kr] = model.predict(state_board)
            move = max(Q.items(), key=operator.itemgetter(1))[0]

        # Store move and board state
        if board.turn:
            white_moves.append(move)
            white_fen_history.append(np.array(state_board, copy=True))
        else:
            black_moves.append(move)
            black_fen_history.append(np.array(state_board, copy=True))

        board.push(move)

    # Assign rewards
    if board.result() == "1-0":
        reward(white_fen_history, white_moves, black_fen_history, black_moves)
    elif board.result() == "0-1":
        reward(black_fen_history, black_moves, white_fen_history, white_moves)

    return board.result()


if __name__ == "__main__":
    print("Training the Deep-Q-Network with multiprocessing.")
    print(f"Number of training games: {training_games}")
    print(f"Steps: {steps}")

    for step in range(steps):
        print(f"\nStep {step + 1}/{steps}")

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(simulate_game, range(training_games))

        # Count winners
        for result in results:
            winners[result] = winners.get(result, 0) + 1

        # Decrease epsilon
        epsilon = max(epsilon - decremental_epsilon, 0.1)

        print(f"Step {step + 1} completed. Winners count: {winners}")

    # Save the trained model and moves mapping
    with open('generalized_moves.json', 'w') as fp:
        json.dump(general_moves, fp)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    print("Training completed and model saved.")