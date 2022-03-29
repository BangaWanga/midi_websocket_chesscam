import numpy as np
import threading
from PyQt5.QtWidgets import QApplication

# ugly increase the scope of imports
import sys
sys.path.append('../')
from util.midi_IO import *
from Client.gui import Gui


class step_sequencer:

    def __init__(self):
        self.bpm = 120
        self.step =0 #0...16
        self.sequences = np.zeros((12, 16), dtype=np.int)
        self.gui = None
        self.midi_io = midi_IO()

    def run_gui(self):

        app = QApplication(sys.argv)
        self.gui = Gui()
        app.exec()

    def run_threaded(self):
        thread = threading.Thread(target=self.run_gui)
        thread1 = threading.Thread(target=self.run)

        thread.setDaemon(True)
        thread1.setDaemon(True)

        thread.start()
        thread1.start()

    def run(self):
        print("Setting up guy...")
        while self.gui == None:
            pass

        while True:
            #check if notch has been activated in gui
            if self.gui.notch !=0:
                self.midi_io.clockTicks += self.gui.notch
                self.gui.notch = 0
            curr_step = self.midi_io.clock()
            if curr_step!= self.step:
                self.play()
        self.midi_io.quit()

    def play(self):
        velocity = 100
        self.step = self.midi_io.step
        #self.gui.step(self.step)
        #ToDo: Check if necessary to save var two times
        timestamp = self.midi_io.get_midi_time()


        # loop through the sequences and create the MIDI events of this step
        midiEvents = []  # collect them in this list, then send all at once
        for i, seq in enumerate(self.sequences):
            if seq[self.step] == 1:
                #print("Playing!!")
                midiEvents.append([[0x80, 36 + i, 0],
                                   timestamp - 5])  # note off for all notes (note 36: C0). Reduce timestamp to make sure note off occurs before next note on.
                midiEvents.append([[0x90, 36 + i, velocity], timestamp])  # note on, if a 1 is set in the respective sequence
                # self.midiOut.note_on(36 + i, velocity)
        self.midi_io.write_midi(midiEvents)  # write the events to the MIDI output port

    def set_new_sequence(self, sequence):
        if self.gui == None:
            return
        self.sequences= sequence
        self.show_log(f"NEW SEQUENCE RECEIVED at time {self.midi_io.get_midi_time()}")

        try:
            self.gui.draw_sequence(sequence)
        except:
            print("no sequence drawn")

    def show_log(self,msg):
        self.gui.show_log(msg)




if __name__=="__main__":
    seq = step_sequencer()

