from sequencer import sequencer
import asyncio
import threading


def main():
    s = sequencer(4)
    s.run()


if __name__ == "__main__":
    main()
