from sequencer import sequencer
import asyncio


async def main():
    s = sequencer(4)
    await asyncio.gather(
        s.run(),
        s.handle_network_connection(),
    )


if __name__ == "__main__":
    main()