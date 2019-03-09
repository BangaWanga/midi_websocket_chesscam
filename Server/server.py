#!/usr/bin/env python

# WS server example
from Server.hub import *
import websockets
import asyncio



hub=Hub()
setup_chesscam = asyncio.ensure_future(hub.chesscam())
loop = asyncio.get_event_loop()
asyncio.get_event_loop().run_until_complete(setup_chesscam)

start_server = websockets.serve(hub.hello, 'localhost', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()