#!/usr/bin/env python

# WS server example

import websockets
import json
from Server.chesscam.chesscam import ChessCam
import asyncio
import config


class Hub:

    def __init__(self):
        # First we setup the chesscam
        self.cam = ChessCam()
        self.connected = False
        self.sequence_ready_to_send = False  # is True if there is a sequence to be sent
        self.new_sequence = None

    async def get_new_sequence(self):
        if input("For Sending press 's'\n") == "s":  # if key 's' is pressed
            if not self.connected:
                print("client disconnected.")
            else:
                self.cam.run(user_trigger=True) #Getting new pictures, could be flagged as well
                if self.cam.new_sequence_captured:
                    self.sequence_ready_to_send = True
                return True

    async def handler(self, websocket, path):
        while True:
            try:
                if self.connected:
                    producer_task = asyncio.ensure_future(self.get_new_sequence())
                    await asyncio.gather(producer_task)
                    if self.sequence_ready_to_send:
                        if not websocket.open:
                            websocket = websockets.connect(config.client_connection)
                            print("Websocket closed")
                            self.connected=False
                            for task in asyncio.Task.all_tasks():
                                task.cancel()
                            return
                        await websocket.send(json.dumps(self.cam.track.sequences.tolist()))

                        self.sequence_ready_to_send = False
                        print("New Sequence sent")
                else:
                    listener_task = asyncio.ensure_future(websocket.recv())
                    message1 = await asyncio.gather(listener_task)

                    if message1[0]==config.client_greeting:
                        print(f"{message1}")
                        await websocket.send(config.server_greeting)
                        self.connected = True

            except websockets.ConnectionClosed:
                print("Websocket closed.")
                self.connected=False

    def chesscam(self):

        while not self.cam.grid_captured:
            self.cam.run()
            if self.cam.grid_captured:
                self.new_sequence = self.cam.track.sequences
                self.sequence_ready_to_send = True
                return True     # we can open the server now


if __name__ == "__main__":

    hub = Hub()
