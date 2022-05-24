import typing

import cv2
from chesscam.cam.camera import Camera
from chesscam.cam.overlay import Overlay
import multiprocessing


class ChessCam:
    def __init__(self):
        self.camera = Camera()
        self.overlay = Overlay(self.camera.get_cam_resolution())    # handle scale and pos differently
        print("Chesscam init finished")
        self.chess_data = None
        self.callibrate_fields = None
        self._get_queue: typing.Optional[multiprocessing.Queue] = None
        self._put_queue: typing.Optional[multiprocessing.Queue] = None

    def __call__(self, *args, **kwargs):
        self._get_queue = args[0]
        self._put_queue = args[1]
        while True:
            self.update()
            #self._send_current_chesspositions()

    def receive_msg(self, frame):
        if self._get_queue.empty():
            return
        ret = self._get_queue.get(block=False)
        if ret is None:
            return
        if ret["event"] == "calibrate":
            positions = [(f[0], f[1]) for f in ret["fields"]]
            selected_colors = [f[2] for f in ret]
            self.overlay.calibrate(frame, positions, selected_colors)
        elif ret["event"] == "get_board_colors":
            print("Get Board Colors")
            self._send_current_chesspositions()

    def update(self, show=False):
        frame = self.camera.capture_frame_from_videostream()
        self.overlay.update_color_values(frame)
        #self.calibrate(frame)
        if show:
            cv2.imshow('computer visions', frame)
        self.process_key_input()
        self.receive_msg(frame)

    def process_key_input(self):
        key = cv2.waitKey(1)
        if key == 113 or key == 27:
            self.quit()
        move_size = 10
        offset = self.overlay.offset
        scale = self.overlay.scale
        match key:
            case 97:
                offset = (offset[0] - move_size, offset[1])
            case 100:
                offset = (offset[0] + move_size, offset[1])
            case 119:
                offset = (offset[0], offset[1] - move_size)
            case 115:
                offset = (offset[0], offset[1] + move_size)
            case 43:
                scale += 0.01
            case 45:
                scale -= 0.01
            case -1:
                pass
        self.overlay.change_drawing_options(offset, scale)

    def quit(self):
        # When everything done, release the capture
        self.camera.cam.release()
        cv2.destroyAllWindows()
        quit()

    def _send_current_chesspositions(self):
        print("Send put queue ", self.overlay.chess_board_values)
        self._put_queue.put(self.overlay.chess_board_values)
        # self.chess_data = multiprocessing.Array("i", self.overlay.chess_board_values.tolist())

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    cam = ChessCam()

    while True:
        cam.update()
