from sequencer import sequencer
from util.chess_game import chess_game

if __name__ == "__main__":
    s = sequencer(4)
    s.run()

    c = chess_game()
    print(c.make_next_move())