import numpy as np
from midi_IO import *
import time
import threading

class step_sequencer:

    def __init__(self):
        self.bpm = 120
        self.new_sequence = False #flag for server to get a new sequence
        self.step =0 #0...16
        self.sequences = np.zeros((12, 16), dtype=np.int)

        self.melody()
        #print(self.sequences)
        self.midi_io = midi_IO()
    def run_threaded(self):
        thread = threading.Thread(target=self.run)
        thread.setDaemon(True)
        thread.start()
        print("Sequencer ??")

    def run(self):
        while True:
            curr_step = self.midi_io.clock()
            if (curr_step!= self.step):
                self.play()
        self.midi_io.quit()
    def play(self):

        velocity = 100

        self.step =self.midi_io.step

        # create timestamp for this step with an option of a random delay
        currentTimeInMs = pygame.midi.time()

        
        self.last_step = self.midi_io.get_midi_time()
        #ToDo: Check if necessary to save var two times
        timestamp = self.last_step
        #timestamp = self.last_step

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
        


    




    def melody(self):
        self.sequences[0][0]= 1
        self.sequences[0][4] = 1
        self.sequences[0][8] = 1
        self.sequences[0][12] = 1

if __name__=="__main__":
    seq = step_sequencer()

