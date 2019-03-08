#!/usr/bin/env python
# WS server that sends messages at random intervals

import websockets
from util.midi_IO import*

count = 0
midi = midi_IO()


async def clock(websocket, path):
    task = asyncio.ensure_future(midi.run())
    #task = asyncio.create_task()
    inner_count = 0

    while True:
        await midi.clock()
        if (inner_count != int(midi.count/4)):
            inner_count = int(midi.count/4)
            await websocket.send(str(inner_count))

    task.cancel()





start_server = websockets.serve(clock, '127.0.0.1', 5678)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()