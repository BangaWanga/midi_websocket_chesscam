from time import sleep

import rtmidi

import config
from sequence import Sequence

empty_sequence = [0,0,1,0,0,0,0,0,1,2,0,0,0,3,0,0]

NOTE_ON = 144
NOTE_OFF = 144 - 16

class sequencer:
    def __init__(self, sequence_count = 4):
        self.init_midi()
        self.sequence = []
        for s in range(0,sequence_count):
            self.sequence.append(Sequence(empty_sequence))
        self.bpm = 120
        self.running = False
        self.midi_clock_index = 1
        self.velocity = 127
        self.midi_off_msgs = []


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
        if message == [248]:
            self.midi_clock_index += 1
            if self.midi_clock_index == 24:
                self.process_output()
                self.midi_clock_index = 0

    def getMSFor16inBpm(self):
        return self.bpm / 60 / 16

    def process_output(self):
        messages = []
        for seq in self.sequence:
            msg = seq.run()
            if msg:
                messages.append(msg)

        # clean duplication
        messages = list(dict.fromkeys(messages))

        self.send_midi_off(messages)

        for msg in messages:
            print(msg)
            self.midiout.send_message(self.get_midi_for_valu(msg))

    def get_midi_for_valu(self, val, midi_cmd = NOTE_ON):
        if val in config.midi_value:
            return [midi_cmd, config.midi_value[val], self.velocity]
        else:
            return None

    def set_sequence(self,sequence, nr):
        self.sequence[nr] = sequence

    def send_midi_off(self, messages):
        for msg in self.midi_off_msgs:
            self.midiout.send_message(self.get_midi_for_valu(msg, NOTE_OFF))

        self.midi_off_msgs = messages


if __name__ == "__main__":
    s = sequencer(2)
    s.run()
