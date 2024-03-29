import asyncio
import websockets
import json
import math
from cam import ChessCam
import cv2
import logging
import sys
from queue import Queue
import typing

debug_queue = Queue()

# check if the command-line argument `debug` was passed to the script
# to activate debug mode
if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    debug = True
else:
    debug = False

connected_clients = set()
cam = ChessCam()
TRUE_COLOR_MODE = True
colors_rgb = ((0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0))

pos_id_to_pos_tuple = lambda pos_id: [math.ceil((pos_id - 7) / 8), pos_id % 8]


def calibrate(positions, color_classes) -> str:
    return cam.calibrate(positions, color_classes)


def get_board_colors() -> typing.Dict[int, int]:
    return {"board_colors": cam.chess_board_values}


def broadcast_chessboard_values():
    if connected_clients:
        chess_board_color_classes = get_board_colors()
        json_response = {
            "event": "board_colors",
            "topic": "sequencer:foyer",
            "payload": chess_board_color_classes,
            "ref": ""}
        websockets.broadcast(connected_clients, json_response)
    else:
        # send_shout_to_debug_interface() put into
        debug_queue.put("No clients connected")


async def debug_loop():
    while True:
        if cv2.waitKey(1) == ord("q"):
            break

        frame = cam.debug_field()
        cv2.imshow('Debug View', frame)

    cv2.destroyWindow('Debug View')


async def connect_to_debug_interface(websocket):
    await websocket.send(json.dumps({
        "event": "phx_join",
        "topic": "sequencer:lobby",
        "payload": {},
        "ref": "sequencers:one"}
    ))


async def send_rgb_to_debug_interface(websocket, payload):
    await websocket.send(json.dumps({
        "event": "sequencer_feedback",
        "topic": "sequencer:lobby",
        "payload": payload,
        "ref": ""}
    ))


async def send_shout_to_debug_interface(websocket, msg: str):
    await websocket.send(json.dumps({
        "event": "shout",
        "topic": "sequencer:lobby",
        "payload": {"msg": msg},
        "ref": "sequencers:one"}
    ))


async def send_color_classes_to_debug_interface(websocket):
    mean_colors = cam.get_color_of_all_fields()
    if mean_colors is None:
        await send_shout_to_debug_interface(websocket, "No color detected")
    else:
        values = cam.chess_board_values
        print(values)
        for pos, col_class in values.items():
            payload = {'color': colors_rgb[col_class], 'padid': pos, 'position': [math.ceil((pos - 7) / 8), pos % 8]}
            await send_rgb_to_debug_interface(websocket, payload)
            print(payload)


async def updated_sequencer_pad(websocket, payload, send_all_fields=True):

    if send_all_fields:
        mean_colors = cam.get_color_of_all_fields()
        if mean_colors is None:
            await send_shout_to_debug_interface(websocket, "No color detected")
        else:
            for pos, c in mean_colors.items():
                payload["color"] = [int(i) for i in c]  # color is feedback color
                payload["position"] = list([int(i) for i in pos])  # color is feedback color
                payload["padid"] = (payload["position"][0] * 8) + payload["position"][1]
                await send_rgb_to_debug_interface(websocket, payload)
    else:
        mean_color = cam.get_color_of_field(payload["position"])
        if mean_color is None:
            await send_shout_to_debug_interface(websocket, "No color detected")
        else:
            payload["color"] = list(mean_color)  # color is feedback color
        await send_rgb_to_debug_interface(websocket, payload)


async def handle_listener(websocket):
    while True:
        # Check if button pressed
        if connected_clients:
            broadcast_chessboard_values()
        try:
            message = await websocket.recv()
            json_request = json.loads(message)
        except websockets.ConnectionClosedOK:
            break
        except Exception as e:
            logging.log(msg=e, level=logging.ERROR)
            continue
        if "event" not in json_request:
            json_response = {
                "event": "shout",
                "topic": "sequencer:foyer",
                "payload": {"msg": "Message does not contain event"},
                "ref": ""}
            await websocket.send(json.dumps(json_response))

        if json_request["event"] == "calibrate":
            payload = json_request["payload"]
            positions = [p['position'] for p in payload]
            color_classes = [p["color"] for p in payload]
            calibrate_msg = calibrate(positions, color_classes)
            json_response = {
                "event": "calibrate_feedback",
                "topic": "sequencer:foyer",
                "payload": {"msg": calibrate_msg},
                "ref": ""}
        elif json_request["event"] == "board_colors":
            chess_board_color_classes = get_board_colors()
            json_response = {
                "event": "board_colors",
                "topic": "sequencer:foyer",
                "payload": chess_board_color_classes,
                "ref": ""}
        elif json_request["event"] == "subscribe":
            connected_clients.update(websocket)
            json_response = {
                "event": "subscription_success",
                "topic": "sequencer:foyer",
                "payload": "",
                "ref": ""}
        else:
            json_response = {
                "event": "shout",
                "topic": "sequencer:foyer",
                "payload": {"msg": f"Unknown event {json_request['event']}"},
                "ref": ""}

        await websocket.send(json.dumps(json_response))


async def handle_debug_events(websocket):
    while True:
        response = await websocket.recv()
        json_response = json.loads(response)

        if json_response["event"] == "load_calibration":
            succ = cam.load_color_samples()
            if succ:
                msg = "Loading samples successful"
            else:
                msg = "Loading samples did not work"
            await send_shout_to_debug_interface(websocket, msg)
        elif json_response["event"] == "save_calibration":
            succ = cam.save_color_samples()
            if succ:
                msg = "Saving samples successful"
            else:
                msg = "Saving samples did not work"
            await send_shout_to_debug_interface(websocket, msg)
        elif json_response["event"] == "toggle_true_colors":
            TRUE_COLOR_MODE = not TRUE_COLOR_MODE
            if TRUE_COLOR_MODE:
                await send_color_classes_to_debug_interface(websocket)
            else:
                await updated_sequencer_pad(websocket, payload)
        elif json_response["event"] == "clear":
            # await send_color_classes_to_debug_interface(websocket)
            payload = json_response["payload"]
            await updated_sequencer_pad(websocket, payload)
        elif json_response["event"] == "updated_sequencerpad":
            # if a pad was clicked, we set the same field to the current color measured by the camera
            payload = json_response["payload"]
            await updated_sequencer_pad(websocket, payload)
        elif json_response["event"] == "calibrate":
            payload = json_response["payload"]  # [{'color': 4, 'padid': 1, 'position': [0, 1]}]
            positions = [p['position'] for p in payload]
            color_classes = [p["color"] for p in payload]
            calibrate_msg = calibrate(positions, color_classes)
            logging.log(msg=calibrate_msg, level=logging.DEBUG)
            print("Calibrate")

            await send_shout_to_debug_interface(websocket, calibrate_msg)
            await send_color_classes_to_debug_interface(websocket)

        else:
            await send_shout_to_debug_interface(websocket, msg=f"Unknown event {json_response['event']}")


async def handle_debug_connection(
        debugger_address: str = 'ws://sequencerinterface.local:4000/sequencersocket/websocket'
):
    try:
        async with websockets.connect(debugger_address) as websocket:
            await connect_to_debug_interface(websocket)
            await handle_debug_events(websocket)
        print("Connection with debugger established")
        while True:
            response = await websocket.recv()
            json_response = json.loads(response)

            if json_response["event"] == "load_calibration":
                succ = cam.load_color_samples()
                if succ:
                    msg = "Loading samples successful"
                else:
                    msg = "Loading samples did not work"
                await send_shout_to_debug_interface(websocket, msg)
            elif json_response["event"] == "save_calibration":
                succ = cam.save_color_samples()
                if succ:
                    msg = "Saving samples successful"
                else:
                    msg = "Saving samples did not work"
                await send_shout_to_debug_interface(websocket, msg)
            elif json_response["event"] == "toggle_true_colors":
                TRUE_COLOR_MODE = not TRUE_COLOR_MODE
                if TRUE_COLOR_MODE:
                    await send_color_classes_to_debug_interface(websocket)
                else:
                    await updated_sequencer_pad(websocket, payload)
            elif json_response["event"] == "clear":
                # await send_color_classes_to_debug_interface(websocket)
                payload = json_response["payload"]
                await updated_sequencer_pad(websocket, payload)
            elif json_response["event"] == "updated_sequencerpad":
                # if a pad was clicked, we set the same field to the current color measured by the camera
                payload = json_response["payload"]
                await updated_sequencer_pad(websocket, payload)
            elif json_response["event"] == "calibrate":
                payload = json_response["payload"]    # [{'color': 4, 'padid': 1, 'position': [0, 1]}]
                positions = [p['position'] for p in payload]
                color_classes = [p["color"] for p in payload]
                calibrate_msg = calibrate(positions, color_classes)
                logging.log(msg=calibrate_msg, level=logging.DEBUG)
                print("Calibrate")

                await send_shout_to_debug_interface(websocket, calibrate_msg)
                await send_color_classes_to_debug_interface(websocket)

            else:
                await send_shout_to_debug_interface(websocket, msg=f"Unknown event {json_response['event']}")
    except Exception as e:
        print(e)


async def main():
    # Handler continues even when one of the tasks fails
    if debug:
        debug_task = asyncio.create_task(debug_loop())
        await debug_task

    await asyncio.gather(
        run_server(),
        handle_debug_connection(),
    )
    # Handler stops when one of the tasks stops
    # server_connection = asyncio.create_task(handle_server(websocket))
    # debug_connection = asyncio.create_task(handle_debug_connection())

    # done, pending = await asyncio.wait(
    #    [server_connection, debug_connection],
    #    return_when=asyncio.FIRST_COMPLETED,
    # )
    # for task in pending:
    #    task.cancel()


async def run_server():
    if debug:
        debug_task = asyncio.create_task(debug_loop())
        await debug_task

    async with websockets.serve(handle_listener, "", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())

    # access with

    # loop = asyncio.run(main())
    # try:
    #     asyncio.ensure_future(main())
    #     loop.run_forever()
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     print("Closing Loop")
    #     loop.close()
