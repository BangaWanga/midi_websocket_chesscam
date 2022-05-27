from sequencer import sequencer
import asyncio
import threading


async def main():
    s = sequencer(4)
    t0 = threading.Thread(target=s.run, args=())
    t1 = threading.Thread(target=s.handle_network_connection, args=())
    t0.start()
    t1.start()
    t0.join()
    t1.join()

if __name__ == "__main__":
    main()
