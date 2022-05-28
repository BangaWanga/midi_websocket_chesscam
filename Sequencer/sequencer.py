from time import sleep
import rtmidi
import math
import config
from sequence import Sequence
import websockets
import json
import asyncio

empty_sequence = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

NOTE_ON = 144
NOTE_OFF = 144 - 16

connection = None


async def connect_to_chesscam(websocket):
    await websocket.send(json.dumps({
                "event": "subscribe",
                "topic": "sequencer:foyer",
                "payload": "",
                "ref": ""}))


class sequencer:
    def __init__(self, sequence_count=4, standalone=False):
        # self.init_midi()
        self.sequence_count = sequence_count
        self.clear_sequencer()
        self.bpm = 120
        self.running = False
        self.midi_clock_index = 1
        self.velocity = 127
        self.midi_off_msgs = []
        self.standalone = standalone
        self.sequence = []  # Rolli ist das okay so?

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

    async def run(self):
        self.running = True
        while self.running and not config.use_midi_clock:
            print("Using internal Clock")
            self.process_output()
            sleep(self.getMSFor16inBpm())

        # self.midiin.set_callback(self.handle_midi_input)
        while True:
            try:
                await self.handle_network_connection()  # runs forever
            except (websockets.exceptions.InvalidMessage, ConnectionRefusedError, Exception):
                print("Could not establish connection to chesscam")
                await asyncio.sleep(1)

    async def handle_network_connection(self, chesscam_adress: str = "ws://localhost:8765"):
        global connection
        async with websockets.connect(chesscam_adress) as websocket: # , ping_interval=10, ping_timeout=100) as websocket:
            await connect_to_chesscam(websocket)
            while True:
                try:
                    msg = await websocket.recv()
                except websockets.exceptions.ConnectionClosedError:
                    print("Connection closed")
                    return
                except Exception as e:
                    print("Something went wrong: ", e)
                    return
                print(msg)
                self.handle_network_input(json.loads(msg))

    def handle_network_input(self, json_message: dict):
        if json_message["event"] == "subscription_success":
            print("subscription_success, Yeah!")
        elif json_message["event"] == "board_colors":
            sequences = [empty_sequence for _ in range(4)]
            if not json_message["payload"]["board_colors"]:
                return
            for col_class, field_id in json_message["payload"]["board_colors"].items():
                sequences[math.floor(field_id / 16)][field_id % 16] = col_class
            for idx, s in enumerate(sequences):
                self.set_sequence(idx, s)
        else:
            print("Unknown event")

    def handle_midi_input(self, event, data=None):
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
                messages.append([sequence_nr,msg])
                self.midiout.send_message(self.get_midi_for_valu(msg, sequence_nr))

        for msg in self.midi_off_msgs:
            self.midiout.send_message(self.get_midi_for_valu(msg[1], msg[0], NOTE_OFF))
        self.midi_off_msgs = messages

    def get_midi_for_valu(self, val, sequence_nr = 0, midi_cmd = NOTE_ON):
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
    loop = asyncio.get_event_loop()
    loop.run_until_complete(s.run())
