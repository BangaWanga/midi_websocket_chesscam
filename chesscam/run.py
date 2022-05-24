import typing
from multiprocessing import Process
from cam import Chesscam, conn
from multiprocessing import Process, Pipe
import asyncio
import websockets
import json


def _runner(func: typing.Callable):
    return func()


def callibrate(fields: typing.List[typing.Tuple[int, int]]):
    pass


def get_board_colors():
    pass


endpoints = {
    "callibrate": callibrate,
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
            print(e)
            req = None

        if req.get("type") in endpoints:
            response = endpoints[req.get("type")]
            if response is not None:
                websocket.send(json.dumps(response))
            else:
                print("Callibrated")


parent_conn, child_conn = Pipe(duplex=True)


p0 = Process(target=Chesscam, args=(child_conn,))
p0.start()


async def main():
    async with websockets.serve(handler, "", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":

    asyncio.run(main())
