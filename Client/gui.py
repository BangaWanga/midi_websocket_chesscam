# !/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QPushButton,
                             QFrame, QApplication, QLabel, QVBoxLayout)
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys



font = QFont("Times", 15, QFont.Bold)

class Gui(QWidget):
    def __init__(self):
        super().__init__()
        self.chess_fields_set = False
        self.notch = 0  # -1, 0 or 1
        self.initUI()
        self.step =0

        #self.showMaximized()

    def initUI(self):
        vbox = QVBoxLayout()
        self.log = QLabel()
        self.log.setText("STRARTING CHESSCAM APP... INITIALIZING MIDI...")
        self.log.setGeometry(1000, 1000, 400, 400)
        self.log.setFont(font)
        self.log.setAlignment(Qt.AlignBottom)
        self.log.size()
        vbox.addWidget(self.log)
        self.setLayout(vbox)
        self.col = QColor(0, 0, 0)
        notch_up = QPushButton('NOTCH +', self)
        notch_up.setCheckable(True)
        notch_up.move(10, 10)

        notch_up.clicked[bool].connect(self.notch_func)

        notch_down= QPushButton('NOTCH -', self)
        notch_down.setCheckable(True)
        notch_down.move(10, 60)

        notch_down.clicked[bool].connect(self.notch_func)



        self.squares =[[QFrame(self) for i in range (8)] for i in range(8)]
        self.draw_chess_board()
        self.setGeometry(500, 500, 280, 170)
        self.setWindowTitle('Toggle button')
        self.show()
    def show_log(self, msg):
        self.log.setText(msg)

    def draw_chess_board(self):
        for line in range(8):
            for row in range(8):
                if not self.chess_fields_set:
                    self.squares[line][row ].setGeometry(150+(row *100), 20+(line*100), 100, 100)
                if (line%2==0 and row %2 ==0 or line%2 ==1 and row%2 ==1):
                    self.set_color(line, row, "White" )
                else:
                    self.set_color(line, row, "Black")

    def draw_sequence(self, sequences):
        self.draw_chess_board() # first we have to reset the whole Chessboard
        for i, seq in enumerate(sequences):
            #print(f"Vars are i: {i}, seq {seq} and sequences {sequences}")
            for j, step in enumerate(seq):
                if seq[j] == 1:
                    multipl = 1
                    if i %3==0:
                        col = "Red"
                    elif i %3==1:
                        col = "Green"
                    else:
                        col = "Blue"
                    line = int(i/3)*2
                    if j>=8:

                        line+=1
                    row = j %8
                    #print(f"line: {line}, row: {row}, col: {col}")
                    self.set_color(line, row, col)

    def set_color(self, line, row, color): #Setting color of chess-fields
        val =255
        if color == "Red":
            self.col.setRed(val)
            self.col.setBlue(0)
            self.col.setGreen(0)
        elif color == "Green":
            self.col.setGreen(val)
            self.col.setRed(0)
            self.col.setBlue(0)
        elif color == "White":
            self.col.setGreen(val)
            self.col.setRed(val)
            self.col.setBlue(val)
        elif color == "Black":
            self.col.setGreen(0)
            self.col.setRed(0)
            self.col.setBlue(0)
        elif color == "Weid":
            self.col.setGreen(50)
            self.col.setRed(150)
            self.col.setBlue(75)
        else:
            self.col.setBlue(val)
            self.col.setGreen(0)
            self.col.setRed(0)

        self.squares[line][row].setStyleSheet("QFrame { background-color: %s }" %
                                  self.col.name())
    def notch_func(self, pressed):
        source = self.sender()
        if pressed:
            val = 255
        else:
            val = 0
        if source.text() == "NOTCH +":
            self.notch += 1
            self.show_log("Going faster with notch")
        elif source.text() == "NOTCH -":
            self.notch -= 1
            self.show_log("Going slower with notch")
    def start_sequencer(self, pressed):
        if pressed:
            self.seq_running = True
    def stop_sequencer(self, pressed):
        if pressed:
            self.seq_running = False

    def step(self, step):
        line =0
        if step >7:
            line +=1
        for i in range(4):
            self.set_color(line+i, step%8, "Weird")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Gui()
    sys.exit(app.exec_())