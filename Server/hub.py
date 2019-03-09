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
        self.flag_sequence = False  # is True if there is a sequence to be sent
        self.new_sequence = None

    async def hello(self, websocket, path):

        await self.chesscam()
        while True:
            try:
                if not self.connected:
                    answer = await websocket.recv()
                    greeting = f"Server connected!"
                    await websocket.send(greeting)
                    print(f"{answer}")
                    self.connected= True
                else:
                    if (self.flag_sequence):

                        await websocket.send(json.dumps(self.new_sequence.tolist()))
                        self.flag_sequence=False
                        print(f"New Sequence has been sent")

            except (websockets.exceptions.ConnectionClosed, concurrent.futures._base.CancelledError, concurrent.futures._base.CancelledError) as e:
                print(f"connection lost")
                self.connected =False
                break
    async def chesscam(self):

        await self.cam.run()
        if (self.cam.send_new_sequence):
            self.new_sequence=self.cam.track.sequences
            self.cam.send_new_sequence =False
            self.flag_sequence=True
        if self.cam.grid_captured: #So we know we can setup the server
            return True

if __name__=="__main__":

    hub = Hub()








