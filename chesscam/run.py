import typing
from multiprocessing import Process
from cam import Chesscam
from multiprocessing import Process, Pipe, Lock, Array, Queue
import asyncio
import websockets
import json

{"event": "calibrate", "fields": [[0, 0, 1], [1, 1, 2]]}
{"event": "get_board_colors"}
msg_q = Queue() # ToDo: Put fields for calibrate to queue directly
q2 = Queue() # ToDo: Put fields for calibrate to queue directly


def _runner(*args):
    _queue_get, _queue_put = args
    cam = Chesscam()
    cam(_queue_get, _queue_put)

# ToDo: Endpoints all send data via msg_q, do it in one function


def calibrate(req: dict):
    global msg_q
    fields = req.get("fields")
    if fields is not None:
        msg_q.put(req)


def get_board_colors(_req: dict):
    global msg_q
    msg_q.put(_req)
    try:
        chess_data = q2.get(block=True, timeout=1)
        return {"board_colors": chess_data}
    except Exception as e:
        print(e)
        return {"board_colors": "Empty"}


endpoints = {
    "calibrate": calibrate,
    "get_board_colors": get_board_colors
}


async def handler(websocket):
    while True:
        try:
            message = await websocket.recv()
            req = json.loads(message)
        except websockets.ConnectionClosedOK:
            break
        except Exception as e:
            req = {}

        if req.get("event") in endpoints:
            response = endpoints[req.get("event")](req)
            if response is not None:
                print(response, type(response))
                await websocket.send(json.dumps(response))
            else:
                print("Calibrated")


async def main():
    async with websockets.serve(handler, "", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    p0 = Process(target=_runner, args=(msg_q, q2))
    p0.start()
    asyncio.run(main())
