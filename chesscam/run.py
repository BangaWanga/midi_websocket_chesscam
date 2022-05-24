from cam import ChessCam
import asyncio
import websockets
import json

# {"event": "calibrate", "fields": [[0, 0, 1], [1, 1, 2]]}
# {"event": "get_board_colors"}

cam = ChessCam()


def calibrate(req: dict):
    global cam
    cam.update(req)


def get_board_colors(req: dict):
    global cam
    cam.update(req)
    return {"board_colors": cam.chess_board_values}


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
    asyncio.run(main())
