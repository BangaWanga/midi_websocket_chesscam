import asyncio
import websockets
from midi_IO import*



def list_parser(str_list):
    str_list= str_list[1:-1]
    str_list = str_list.split("[")
    result =[]
    for elem in str_list:
        if (len(elem)>2):
            elem = elem.replace("[", "")
            elem = elem.replace("]", "")
            elem = elem.replace(" ", "")
            elem = [int(i) for i in elem.split(",") if i !="" ]
            result.append([[elem[0], elem[1], elem[2], elem[3]], elem[4]])
    return result



async def send_midi_data():
    async with websockets.connect(
            'ws://localhost:8765') as websocket:
        while True:
            midiio.send_midi( await websocket.recv())


async def get_tempo():
    async with websockets.connect(
            'ws://localhost:8765') as websocket:
        while True:
            greeting = await websocket.recv()
            print(greeting)


#[[[[248, 0, 0, 0], 15078], [[128, 66, 64, 0], 15078], [[144, 67, 1, 0], 15078], [[144, 66, 1, 0], 15078]]]

#print(list_parser("[[[[248, 0, 0, 0], 15078], [[128, 66, 64, 0], 15078], [[144, 67, 1, 0], 15078], [[144, 66, 1, 0], 15078]]]"), "RESULT")
successful = False
midiio = midi_IO()
while not successful:
    try:
        asyncio.get_event_loop().run_until_complete(send_midi_data())
        successful = True
    except:

        pass
asyncio.get_event_loop().run_forever()

