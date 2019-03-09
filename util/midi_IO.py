import pygame
import pygame.midi
import asyncio
import config


class midi_IO(asyncio.Task):
    def __init__(self, pygame_midi_enabled =True):
        self.pygame_midi_enabled = pygame_midi_enabled
        if (pygame_midi_enabled):
            pygame.init()
            pygame.midi.init()
            self.midiOut = self.ask_for_midi_device(kind="output",default_value=config.midiout)
            self.midiIn = self.ask_for_midi_device(kind="input", default_value=config.midiin)

        # initialize state
        self.step = 0  # current step based on clock sync
        self.clockTicks = 0  # counter for received clock ticks
        self.last_timestamp = 0 #Last timestamp of a new step
        self.running = True
        self.randomness = 0.

    async def read(self, number):
        if self.pygame_midi_enabled:
            return self.midiIn.read(number)
    async def send_midi(self, midi_notes): #note = [[[[248, 0, 0, 0], 159400]]] -> [Command, Channel, Velocity, ?], Timestamp]
        if self.pygame_midi_enabled:
            self.midiOut.write(midi_notes)

    def clock(self):

        for midiEvent in self.midiIn.read(15):  # read 5 MIDI events from the buffer. TODO: Good number?
            if (midiEvent[0][0]) == 248:
                self.clockTicks = (self.clockTicks + 1) % 6  # count the clock ticks
                self.last_timestamp = midiEvent[1]
                if (self.clockTicks == 0):  # 12 clock ticks are one 16th note
                    self.clockTicks = 0  # reset the tick counter
                    self.step = (self.step + 1) % 16  # advance the 16th note counter

                    
        return self.step

    def write_midi(self, midi):

        self.midiOut.write(midi)
        
    def quit(self):
        if self.pygame_midi_enabled:
            self.midiOut.close()
            self.midiIn.close()
        pygame.quit()
    def get_midi_time(self):
        return pygame.midi.time()

    def ask_for_midi_device(self, kind="input", default_value=False):

        if (default_value != False):
            return self.__return_pygame_IO__(kind, default_value)

        """ Let the user choose the midi device to use """
        # Check, if we are looking for a valid kind:
        assert (kind in ("input", "output")), "Invalid MIDI device kind: {0}".format(kind)

        is_input = (kind == "input")
        # First, print info about the available devices:
        print()
        print("Available MIDI {0} devices:".format(kind))
        device_ids = []  # list to hold all available device IDs
        device_id = 0
        info_tuple = ()
        # print info about all the devices
        while not pygame.midi.get_device_info(device_id) is None:
            info_tuple = pygame.midi.get_device_info(device_id)
            if info_tuple[2] == is_input:  # this holds 1 if device is an input device, 0 if not
                print("ID: {0}\t{1}\t{2}".format(device_id, info_tuple[0], info_tuple[1]))
                device_ids.append(device_id)
            device_id += 1
        assert (device_id > 0), "No {0} devices found!".format(kind)

        user_input_id = -1
        while not user_input_id in device_ids:  # ask for the desired device ID until one of the available ones is given
            user_input = input("Which device would you like to use as {0}? Please provide its ID: ".format(kind))
            try:  # try to cast the user input into an int
                user_input_id = int(user_input)
            except ValueError:  # if it fails because of incompatible input, let them try again
                pass
        info_tuple = pygame.midi.get_device_info(user_input_id)
        print(
            "Chosen {0} device: ID: {1}\t".format(kind, user_input_id) + str(info_tuple[0]) + "\t" + str(info_tuple[1]))
        # Open port from chosen device
        if kind == "input":
            return pygame.midi.Input(device_id=user_input_id)
        elif kind == "output":
            return pygame.midi.Output(device_id=user_input_id, latency=0)

    def __return_pygame_IO__(self, kind, user_input_id):
        if kind == "input":
            return pygame.midi.Input(device_id=user_input_id)
        elif kind == "output":
            return pygame.midi.Output(device_id=user_input_id, latency=0)

if __name__=="__main__":
    midi = midi_IO()

    midi.quit()