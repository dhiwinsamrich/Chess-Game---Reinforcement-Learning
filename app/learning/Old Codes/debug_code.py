import chess
import random
import json
import operator
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import logging
import argparse
import os
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create network. Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
model.add(Dense(20, input_shape=(65,), kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))  # Output layer
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

np.set_printoptions(threshold=np.inf)
state_board = np.zeros((1, 65))  # 64 squares + selected move index

# Value of every piece
piece_values = {
    'p': 10, 'P': -10, 'q': 90, 'Q': -90, 'n': 30, 'N': -30,
    'r': 50, 'R': -50, 'b': 30, 'B': -30, 'k': 900, 'K': -900, None: 0
}

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--number_of_games', type=int, default=100)
parser.add_argument('--winner_reward', type=float, default=1)
parser.add_argument('--loser_malus', type=float, default=-1)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--decremental_epsilon', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.05)
args = parser.parse_args()

# Parameters
training_games = args.number_of_games
winner_reward = args.winner_reward
loser_malus = args.loser_malus
epsilon = args.epsilon
decremental_epsilon = args.decremental_epsilon
gamma = args.gamma
general_moves = {}
winners = {}  # Win counts
steps = 1000

logging.info("Training parameters:")
logging.info(f"Number of games: {training_games}, Winner reward: {winner_reward}, Loser malus: {loser_malus}, Epsilon: {epsilon}, Gamma: {gamma}")

# Helper functions
def evaluate_board(turn):
    """Evaluate the board and update state_board."""
    total = 0
    mult = 1 if turn else -1
    for i, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        total += mult * piece_values.get(str(piece), 0)
        state_board[0][i] = mult * piece_values.get(str(piece), 0)
    return total

def get_int(move):
    """Map a move to an index in general_moves."""
    if str(move) not in general_moves:
        general_moves[str(move)] = len(general_moves)
    return general_moves[str(move)]

def reward(fen_history, moves, lose_fen, lose_moves):
    """Calculate rewards for the winner and loser."""
    try:
        for i, state in enumerate(fen_history):
            gamma_weight = gamma * (1 / len(fen_history))
            state[0][64] = get_int(moves[i])
            target = model.predict(state) + winner_reward * gamma_weight
            model.train_on_batch(state, target)
        for i, state in enumerate(lose_fen):
            gamma_weight = gamma * (1 / len(lose_fen))
            state[0][64] = get_int(lose_moves[i])
            target = model.predict(state) + loser_malus * gamma_weight
            model.train_on_batch(state, target)
    except Exception as e:
        logging.error(f"Error during reward calculation: {e}")

def save_model():
    """Save the trained model and move mappings."""
    try:
        model.save("model.h5")
        with open("generalized_moves.json", "w") as f:
            json.dump(general_moves, f)
        logging.info("Model and move mappings saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def signal_handler(sig, frame):
    """Handle keyboard interrupt."""
    logging.warning("Keyboard interrupt detected. Saving model...")
    save_model()
    sys.exit(0)

# Signal handler for keyboard interrupts
signal.signal(signal.SIGINT, signal_handler)

# Training loop
for joum in range(steps):
    for i in range(training_games):
        board = chess.Board()
        fen_history, black_moves, white_moves = [], [], []
        black_fen_history, white_fen_history = [], []
        evaluation_history = []

        while not board.is_game_over():
            # Choose a move
            if np.random.rand() <= epsilon:
                move = random.choice(list(board.legal_moves))
            else:
                evaluate_board(board.turn)
                Q = {}
                for legal_move in board.legal_moves:
                    state_board[0][64] = get_int(legal_move)
                    Q[legal_move] = model.predict(state_board)
                move = max(Q.items(), key=operator.itemgetter(1))[0]

            # Record the state and move
            fen_history.append(np.copy(state_board))
            (white_moves if board.turn else black_moves).append(str(move))
            (white_fen_history if board.turn else black_fen_history).append(np.copy(state_board))

            # Make the move
            try:
                board.push(move)
            except ValueError as e:
                logging.error(f"Illegal move {move}: {e}")
                break

        # Reward the winner
        if board.result() == "1-0":
            reward(fen_history, white_moves, black_fen_history, black_moves)
            winners["White"] = winners.get("White", 0) + 1
        elif board.result() == "0-1":
            reward(fen_history, black_moves, white_fen_history, white_moves)
            winners["Black"] = winners.get("Black", 0) + 1
        else:
            logging.info("Game ended in a draw.")

        epsilon = max(epsilon - decremental_epsilon, 0.01)  # Decay epsilon

    # Save model periodically
    if joum % 100 == 0:
        save_model()

# Final save
save_model()
