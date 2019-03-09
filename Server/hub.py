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
                    user_trigger =False

                    if input("For Sending press 's'\n") =="s":  # if key 's' is pressed
                        self.sequence_ready_to_send=True
                        user_trigger=True
                    self.cam.run(user_trigger) #Getting new pictures, could be flagged as well


                    if self.sequence_ready_to_send:
                        await websocket.send(json.dumps(self.cam.track.sequences.tolist()))
                        self.sequence_ready_to_send=False
                        print(f"New Sequence has been sent")

            except Exception as e:
                print(f"connection lost because of Error: {type(e).__name__}")
                self.connected =False
                break


    def chesscam(self):

        while not self.cam.grid_captured:
            self.cam.run()
            if self.cam.grid_captured:

                self.new_sequence=self.cam.track.sequences
                self.sequence_ready_to_send=True


                return True #we can open the server now

if __name__=="__main__":

    hub = Hub()








