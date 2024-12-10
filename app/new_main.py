from flask import Flask, render_template, send_from_directory, request, json
from keras.models import model_from_json
import chess
import numpy as np
import operator

app = Flask(__name__)

# Load the saved model and weights
json_file = open('D:\\III - Semester\\Mini Project\\Deep-Learning\\RL - Chess Game\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("D:\\III - Semester\\Mini Project\\Deep-Learning\\RL - Chess Game\\model.h5")
loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Load generalized moves
json_file = open('D:\\III - Semester\\Mini Project\\Deep-Learning\\RL - Chess Game\\generalized_moves.json')
json_str = json_file.read()
generalized_moves = json.loads(json_str)

# State representation
state_board = np.zeros((1, 65))
switch = {
    'p': 10, 'P': -10,
    'q': 90, 'Q': -90,
    'n': 30, 'N': -30,
    'r': 50, 'R': -50,
    'b': 30, 'B': -30,
    'k': 900, 'K': -900,
    'None': 0
}
squares = chess.SQUARES

@app.route('/')
def index():
    return render_template("reinforced.html")

@app.route('/minimax')
def minimax():
    return render_template('minimax-alpha-beta.html')

@app.route('/random')
def random():
    return render_template('random.html')

@app.route('/img/<path:path>')
def send_js(path):
    return send_from_directory('static/img', path)

@app.route('/send_move', methods=['POST'])
def sender_moves():
    if request.method == "POST":
        fen = request.json['fen']
        turn = request.json['turn']
        board = chess.Board(fen=fen)

        # Set turn based on input
        board.turn = True if str(turn) != "b" else False

        Q = {}
        mult = 1 if board.turn else -1
        a = 0
        for i in squares:
            state_board[0][a] = mult * switch.get(str(board.piece_at(i)), 0)
            a += 1
        for i in board.legal_moves:
            state_board[0][64] = generalized_moves[str(i)]
            Q[str(i)] = loaded_model.predict(state_board)[0][0]

        best_move = max(Q.items(), key=operator.itemgetter(1))[0]
        print(f"Best move is: {best_move}")
        return json.dumps(best_move), 200, {'ContentType': 'application/json'}

if __name__ == '__main__':
    app.run()
