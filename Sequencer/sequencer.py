from time import sleep

import rtmidi

import config
from Sequencer.util.chess_game import chess_game
from sequence import Sequence

empty_sequence = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

NOTE_ON = 144
NOTE_OFF = 144 - 16

class sequencer:
    def __init__(self, sequence_count = 4):
        self.init_midi()
        self.sequence_count = sequence_count
        self.clear_sequencer()
        self.bpm = 120
        self.running = False
        self.midi_clock_index = 1
        self.velocity = 127
        self.midi_off_msgs = []

    def clear_sequencer(self):
        self.sequence = []
        for s in range(0, self.sequence_count):
            self.sequence.append(Sequence(empty_sequence))

    def init_midi(self):
        self.midiout = rtmidi.MidiOut()
        self.midiin = rtmidi.MidiIn()
        available_ports = self.midiin.get_ports()
        print(available_ports)

        # if config Port set use ist
        if config.midi_virtual_channel:
            self.midiout.open_virtual_port(config.midi_virtual_channel_name)
            self.midiin.open_virtual_port(config.midi_virtual_channel_name)
        else:
            self.midiout.open_port(available_ports.index(config.midi_default_out))
            self.midiin.open_port(available_ports.index(config.midi_default_in))

        self.midiin.ignore_types(sysex=True,
                             timing=False,
                             active_sense=True)

    def run(self):
        self.running = True

        while (self.running and not config.use_midi_clock):
            self.process_output()
            sleep(self.getMSFor16inBpm())

        self.midiin.set_callback(self.handle_input)

        while True:
            pass

    def handle_input(self, event, data=None):
        message, deltatime = event
        # tirck
        if message == [248]:
            self.midi_clock_index += 1
            if self.midi_clock_index == 6:
                self.midi_clock_index = 0
                self.process_output()
        # start and contiunue
        if message == [250] or message ==[251]:
            self.midi_clock_index += 1
            self.process_output()
        # stop
        if message == [252]:
            self.midi_clock_index = 0
            self.clear_sequencer()

    def getMSFor16inBpm(self):
        return self.bpm / 60 / 16

    def process_output(self):
        messages = []
        for sequence_nr, seq in enumerate(self.sequence):
            msg = seq.run()
            if msg:
                messages.append(msg)
                self.midiout.send_message(self.get_midi_for_valu(msg, sequence_nr))


    def get_midi_for_valu(self, val, sequence_nr = 0, midi_cmd = NOTE_ON):
        print(midi_cmd + sequence_nr)
        if val in config.midi_value[sequence_nr]:
            midi_val = config.midi_value[sequence_nr][val]
            return [midi_cmd + sequence_nr, midi_val , self.velocity]
        else:
            return None

    def set_sequence(self,nr, sequence):
        self.sequence[nr] = sequence

    def send_midi_off(self, messages):
        for msg in self.midi_off_msgs:
            self.midiout.send_message(self.get_midi_for_valu(msg, NOTE_OFF))

        self.midi_off_msgs = messages


if __name__ == "__main__":
    s = sequencer(4)

    c = chess_game("util/Fischer.pgn")
    seq = c.play_all()

    sequenc_number = 40
    s.set_sequence(0, Sequence(seq[sequenc_number][0]))
    s.set_sequence(1, Sequence(seq[sequenc_number][1]))
    s.set_sequence(2, Sequence(seq[sequenc_number][2]))
    s.set_sequence(3, Sequence(seq[sequenc_number][3]))

    s.run()
