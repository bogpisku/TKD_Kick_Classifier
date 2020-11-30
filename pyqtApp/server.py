import asyncio
import websockets
import time

current_milli_time = lambda: int(round(time.time() * 1000))
async def handle_message(message):
    print(message)

async def consumer_handler(websocket, path):
    while True:
        message = await websocket.recv()
        await handle_message(message)

start_server = websockets.serve(consumer_handler, '', 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
