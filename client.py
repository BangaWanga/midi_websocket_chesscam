import websockets
from sequencer import*
import numpy as np
import concurrent.futures
import json

seq = step_sequencer()
connected=False

async def get_new_sequences():

    global connected
    print("Trying to connect...")
    while not connected:
        try:
            async with websockets.connect(
                    'ws://localhost:8765') as websocket:
                while True:
                    if not connected:
                        await websocket.send(f"Client connected")
                        greeting = await websocket.recv()
                        print(f"< {greeting}")
                        connected = True
                    else:
                        sequence = np.asarray(json.loads(await websocket.recv()), dtype=np.int)
                        seq.set_new_sequence(sequence)

        except (websockets.exceptions.ConnectionClosed, concurrent.futures._base.CancelledError, OSError, ConnectionResetError, json.decoder.JSONDecodeError) as e:
            try:
                seq.show_log("Connection closed because of Error: " + type(e).__name__)
                if type(e).__name__ == 'JSONDecodeError':
                    connected = False
            except:
                pass

async def run_sequencer():
    seq.run_threaded()



if __name__ == '__main__':

    
    event_loop = asyncio.get_event_loop()

    try:
        asyncio.ensure_future(get_new_sequences())
        asyncio.ensure_future(run_sequencer())
        #event_loop.run_until_complete(get_new_sequences())
        event_loop.run_forever()
    except :
        pass


    #event_loop.run_until_complete(task)
    #event_loop.run_until_complete(task1)
    #event_loop.run_forever()
    finally:
        event_loop.close()