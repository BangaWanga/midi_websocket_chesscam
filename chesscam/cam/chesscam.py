import cv2
from cam.camera import Camera
from cam.overlay import Overlay
from cam.image_processing import get_board_parameters, standardize_position


class ChessCam:
    def __init__(self):
        self.camera = Camera()
        self.resolution = self.camera.get_cam_resolution()
        self.padding = 5
        self.board_params = get_board_parameters((500, 500), self.padding)
        offset, field_width, field_height = self.board_params
        self.overlay = Overlay(self.resolution, offset, field_width, field_height)    # handle scale and pos differently

    @property
    def chess_board_values(self) -> dict:
        return self.overlay.chess_board_values

    def update(self, message=None):
        frame = self.camera.capture_frame_from_videostream()
        frame_std, _, _ = standardize_position(frame, self.resolution, self.padding, debug='')

        if frame_std is None:
            print("No field detected")
            return
        if message:
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


if __name__== "__main__":
    c= ChessCam()
    while True:
        c.update(show=True)
