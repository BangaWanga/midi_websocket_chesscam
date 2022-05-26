from cam import ChessCam
import asyncio
import websockets
import json
import sys
import cv2

# {"event": "calibrate", "fields": [[7, 7, 2]]}
# {"event": "get_board_colors"}

# check if the command-line argument `debug` was passed to the script
# to activate debug mode
if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    debug = True
else:
    debug = False

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


async def debug_loop():
    while True:
        if cv2.waitKey(1) == ord("q"):
            break

        frame = cam.debug_field()
        cv2.imshow('Debug View', frame)

    cv2.destroyWindow('Debug View')


async def main():
    if debug:
        debug_task = asyncio.create_task(debug_loop())
        await debug_task

    async with websockets.serve(handler, "", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
