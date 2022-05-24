import cv2
from cam.camera import Camera
from cam.overlay import Overlay
from cam.image_processing import get_board_parameters, standardize_position


class ChessCam:
    def __init__(self):
        self.camera = Camera()
        self.overlay = Overlay(self.camera.get_cam_resolution())    # handle scale and pos differently
        print("Chesscam init finished")
        self.board_params = get_board_parameters()

    @property
    def chess_board_values(self) -> dict:
        return self.overlay.chess_board_values

    def receive_msg(self, message, frame):
        if message["event"] == "calibrate":
            print("Calibrate ", message)
            positions = [(f[0], f[1]) for f in message["fields"]]
            selected_colors = [f[2] for f in message["fields"]]
            self.overlay.calibrate(frame, positions, selected_colors)
        elif message["event"] == "get_board_colors":
            print("Get Board Colors")

    def update(self, message):
        frame = self.camera.capture_frame_from_videostream()
        frame_std, _, _ = standardize_position(frame, debug='')
        if frame_std is None:
            return
        if message["event"] == "calibrate":
            positions = [(f[0], f[1]) for f in message["fields"]]
            selected_colors = [f[2] for f in message["fields"]]
            self.overlay.calibrate(frame_std, positions, selected_colors)
        elif message["event"] == "get_board_colors":
            self.overlay.update_color_values(frame_std)

    def quit(self):
        # When everything done, release the capture
        self.camera.cam.release()
        cv2.destroyAllWindows()
        quit()


if __name__ == "__main__":
    cam = ChessCam()

    while True:
        cam.update()
