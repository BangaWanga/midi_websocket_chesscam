#!/usr/bin/env python

# WS server example
from Server.hub_alt import *
import websockets
import asyncio



hub=Hub()


hub.chesscam()


def custom_exception_handler(loop, context):
    # first, handle with default handler
    loop.default_exception_handler(context)

    exception = context.get('exception')

    print(context)
    loop.stop()


loop = asyncio.get_event_loop()
loop.set_exception_handler(custom_exception_handler)
start_server = websockets.serve(hub.handler, 'localhost', 8765, ping_interval=70)




loop.run_until_complete(start_server)
loop.run_forever()