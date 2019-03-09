#!/usr/bin/env python

# WS server example

import websockets
import concurrent
import json
from Server.chesscam import ChessCam
import asyncio



class Hub:

    def __init__(self):
        # First we setup the chesscam
        self.cam = ChessCam()
        self.connected = False
        self.sequence_ready_to_send = False  # is True if there is a sequence to be sent
        self.new_sequence = None

    async def hello(self, websocket, path):

        while True:
            try:
                if not self.connected:
                    answer = await websocket.recv()
                    greeting = f"Server connected!"
                    await websocket.send(greeting)
                    print(f"{answer}")
                    self.connected= True
                else:
                    self.cam.run()
                    if (self.sequence_ready_to_send):

                        await websocket.send(json.dumps(self.new_sequence.tolist()))
                        self.sequence_ready_to_send=False
                        print(f"New Sequence has been sent")

            except (websockets.exceptions.ConnectionClosed, concurrent.futures._base.CancelledError, concurrent.futures._base.CancelledError) as e:
                print(f"connection lost")
                self.connected =False
                break
    def chesscam(self):

        while not self.cam.new_sequence_captured:
            self.cam.run()
            if self.cam.grid_captured:

                self.new_sequence=self.cam.track.sequences
                self.sequence_ready_to_send=True
                print(self.new_sequence)
        if self.cam.new_sequence_captured: #So we know we can setup the server
            print("It worked!")
            self.cam.new_sequence_captured=False
            return True

if __name__=="__main__":

    hub = Hub()








