import cv2
from cam import camera
from cam import overlay
from cam import image_processing


class ChessCam:
    def __init__(self):
        self.camera = camera.Camera()
        self.resolution = self.camera.get_cam_resolution()
        self.padding = 5
        self.board_params = image_processing.get_board_parameters((500, 500), self.padding)
        offset, field_width, field_height = self.board_params
        self.overlay = overlay.Overlay(self.resolution, offset, field_width, field_height)    # handle scale and pos differently

    @property
    def chess_board_values(self) -> dict:
        return self.overlay.chess_board_values

    def get_frame(self, debug=False):
        frame = self.camera.capture_frame_from_videostream()
        frame_std, _, _ = image_processing.standardize_position(frame, (500, 500), self.padding, debug='')

        if frame_std is None and debug:
            return frame

        return frame_std

    def update(self, message=None):
        frame_std = self.get_frame()

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

    def debug_field(self):
        frame_std = self.get_frame(debug=True)

        frame_std = self.overlay.draw_rect(frame_std.copy())
        return frame_std


if __name__== "__main__":
    c = ChessCam()
    while True:
        ret = c.debug_field()
        if cv2.waitKey(1) == ord("q"):
            break
        if ret is not None:
            cv2.imshow("Name", ret)
