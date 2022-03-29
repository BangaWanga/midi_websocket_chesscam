from apscheduler.schedulers.background import BackgroundScheduler
import time
import atexit
from Server.chesscam import ChessCam


class Hub:
    def __init__(self, cam_interval_seconds=1.):
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(func=self.poll_chesscam, trigger="interval", seconds=cam_interval_seconds)
        # Shut down the scheduler when exiting the app
        atexit.register(lambda: self.scheduler.shutdown())
        self.chesscam = ChessCam()

    def poll_chesscam(self):
        self.chesscam.run(user_trigger=True)

    def start(self):
        print("Starting scheduler")
        self.scheduler.start()

    @property
    def chessboard_state(self):
        """
        returns current state of chessboard
        """
        return self.chesscam.states
