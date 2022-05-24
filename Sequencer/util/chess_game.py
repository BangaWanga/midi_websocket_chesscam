import chess.pgn

party = """1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Bd3 Bg4 7.O-O Nc6 8.Be3 Nd7 
9.Be2 Bxf3 10.Bxf3 e5 11.d5 Ne7 12.Be2 f5 13.f4 h6 14.Bd3 Kh7 15.Qe2 fxe4
16.Nxe4 Nf5 17.Bd2 exf4 18.Bxf4 Ne5 19.Bc2 Nd4 20.Qd2 Nxc4 21.Qf2 Rxf4 22.Qxf4 Ne2+
23.Kh1 Nxf4  0-1"""


class chess_game:
    def __init__(self, party = "Fischer.pgn"):
        self.game = chess.pgn.read_game(open(party))

        self.board = self.game.board()
        self.moves = []
        for m in self.game.mainline_moves():
            self.moves.append(str(m))

        self.index = 0

    #@classmethod
    def play_all(self):
        sequences = []
        for move in self.game.mainline_moves():
            self.board.push(move)
            sequences_str = self.pare_board().split("\n")
            sequences.append(self.get_sequencer_line(sequences_str))

        return sequences

    def make_next_move(self):
        self.index = (self.index + 1) % 20

        move = self.moves[self.index]
        sequences_str = self.pare_board().split("\n")
        return self.get_sequencer_line(sequences_str)


    def pare_board(self):
        board_str = str(self.board).lower()
        board_str = board_str.replace(".", "0")
        board_str = board_str.replace("p", "1")
        board_str = board_str.replace("n", "2")
        board_str = board_str.replace("b", "3")
        board_str = board_str.replace("r", "4")
        board_str = board_str.replace("q", "5")
        board_str = board_str.replace("q", "6")
        board_str = board_str.replace("k", "7")
        return board_str

    def get_sequencer_line(self, sequences_str):
        sequences = []
        for i in range(0, 4):
            new_sequence = sequences_str[i].split(" ") + sequences_str[i + 4].split(" ")
            sequences.append(new_sequence)
        return sequences


if __name__ == "__main__":
    c = chess_game()
    seq =c.play_all()

    for s in seq:
        print(s)


