import asyncio
import typing
import websockets
import json
from concurrent.futures import ProcessPoolExecutor


def callibrate(fields: typing.List[typing.Tuple[int, int]]):
    pass


def get_board_colors():
    pass


endpoints = {
    "calibrate": callibrate,
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
                print("Calibrated")


async def main():
    async with websockets.serve(handler, "", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    executor = ProcessPoolExecutor(2)
    loop = asyncio.get_event_loop()
    boo = loop.run_in_executor(executor, main)

    loop.run_forever()
