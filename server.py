from midi_IO import *
import asyncio
import websockets
import numpy as np

midi_count= 5 #number of notes that are transmitted


async def midi_in(websocket, path):
    while True:
        #midi_data = await midi.read(midi_count)
        #midi_data = lambda midi_data9 : (n for n in midi_data)
        result = str(await asyncio.gather(midi.read(midi_count)))

        if (len(result)) > 4: #string length of empty midi-note
            await websocket.send(result)
        break



async def clock(websocket, path):
    #task = asyncio.ensure_future(midi.run())
    #task = asyncio.create_task()
    inner_count = 0

    while True:
        await midi.clock()
        if (inner_count != int(midi.count/2)):
            inner_count = int(midi.count/2)
            print("Sending" + str(inner_count))
            await websocket.send(str(inner_count))





midi = midi_IO()

start_server = websockets.serve(midi_in, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()




