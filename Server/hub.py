from apscheduler.schedulers.background import BackgroundScheduler
import time
import atexit
from Server.chesscam.chesscam import ChessCam


class Hub:
    def __init__(self, cam_interval_seconds=1.):
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=self.poll_chesscam, trigger="interval", seconds=cam_interval_seconds)
        # Shut down the scheduler when exiting the app

        atexit.register(lambda: scheduler.shutdown())
        self.chesscam = ChessCam()
        scheduler.start()

    def poll_chesscam(self):
        self.chesscam.run(user_trigger=True)

    def start(self):
        print("Starting scheduler")

    @property
    def chessboard_state(self):
        """
        returns current state of chessboard
        """
        return self.chesscam.states


if __name__ == '__main__':
    hub = Hub(1)
    for i in range(10):
        time.sleep(1)