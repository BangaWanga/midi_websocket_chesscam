from sequencer import sequencer
import asyncio


if __name__ == "__main__":
    s = sequencer(4)
    asyncio.run(s.run())
