import websockets
import concurrent.futures
import json
import numpy as np
from Client.sequencer import *
import asyncio


class Client:
    def __init__(self):
        self.seq = step_sequencer()
        self.connected = False

    async def get_new_sequences(self):

        print("Trying to connect...")
        while not self.connected:
            try:
                async with websockets.connect(
                        'ws://localhost:8765') as websocket:
                    while True:
                        if not self.connected:
                            await websocket.send(config.client_greeting)
                            msg = await websocket.recv()
                            if msg == config.server_greeting:
                                print(msg)
                                self.connected = True
                        else:
                            sequence = np.asarray(json.loads(await websocket.recv()), dtype=np.int)
                            self.seq.set_new_sequence(sequence)
                            print(f"new sequence: {sequence}")
            except:
                pass

#            websockets.exceptions.ConnectionClosed, concurrent.futures._base.CancelledError, OSError, ConnectionResetError,
 #           json.decoder.JSONDecodeError) as e:
 #                try:
 #                    seq.show_log("Connection closed because of Error: " + type(e).__name__)
 #                    if type(e).__name__ == 'JSONDecodeError':
 #                        connected = False



    async def run_sequencer(self):
        self.seq.run_threaded()


if __name__ == '__main__':

    event_loop = asyncio.get_event_loop()

    try:
        client = Client()

        asyncio.ensure_future(client.get_new_sequences())
        asyncio.ensure_future(client.run_sequencer())
        event_loop.run_forever()
    except:
        pass
    finally:
        event_loop.close()