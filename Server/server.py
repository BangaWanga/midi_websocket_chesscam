from flask import Flask, redirect
from Server import chesscam
from hub import Hub

app = Flask(__name__)

hub = Hub()


@app.route("/")
def index():
    hub.start()
    return "Chesscam!!!"


@app.route("/get_chessboard")
def get_chessboard():
    chess_state = hub.chessboard_state.tolist()
    print("State! ", chess_state, type(chess_state))
    return {"chessboard": chess_state}


if __name__ == "__main__":
    app.run(debug=True)