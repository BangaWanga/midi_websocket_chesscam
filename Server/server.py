#!/usr/bin/env python

# WS server example

import asyncio
import websockets
import numpy as np
import random
import concurrent
import json

connected = False

async def hello(websocket, path):

    while True:
        global connected
        try:
            if not connected:
                answer = await websocket.recv()
                greeting = f"Server connected!"
                await websocket.send(greeting)
                print(f"{answer}")
                connected= True

            sequence = np.zeros((12, 16), dtype=np.int)
            for i in range(30):
                sequence[random.randint(0, 11)][random.randint(0, 15)] = random.randint(0, 1)

            #await websocket.send(f""+str(sequence))
            await websocket.send(json.dumps(sequence.tolist()))
            print(f"New Sequence has been sent")
            await asyncio.sleep(8)
        except (websockets.exceptions.ConnectionClosed, concurrent.futures._base.CancelledError, concurrent.futures._base.CancelledError) as e:
            print(f"connection lost")
            connected =False
            break



async def send_sth(websocket, path):
    await print("Test")

start_server = websockets.serve(hello, 'localhost', 8765, ping_timeout=50)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()




