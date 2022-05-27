import typing
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
    def chess_board_values(self) -> typing.Dict[int, int]:
        return self.get_board_colors()

    def get_frame(self, debug=False):
        frame = self.camera.capture_frame_from_videostream()

        target_img_wh = (500, 500)
        frame_std, _, _ = image_processing.standardize_position(frame, (500, 500), self.padding, debug='')

        if frame_std is not None:
            origin, field_w, field_h = get_board_parameters(target_img_wh, self.padding)
            frame_std = image_processing.balance_colors(frame_std, origin, field_w, field_h)

        if frame_std is None and debug:
            return frame

        return frame_std

    def get_color_of_field(self, field: typing.Tuple[int, int]) -> typing.Optional[typing.Tuple[int, int, int]]:
        frame_std = self.get_frame()
        if frame_std is None:
            return None
        #   self.overlay.update_color_values(frame_std)
        return self.overlay.get_square_color(frame_std, field)

    def get_color_of_all_fields(self): # -> typing.Optional[typing.List[typing.Dict[typing.Tuple[int, int]], typing.Tuple[int, int, int]]]:
        frame_std = self.get_frame()
        if frame_std is None:
            return None
        colors = {tuple(pos): self.overlay.get_square_color(frame_std, pos) for pos in self.overlay.grid_positions}
        return colors

    def save_color_samples(self) -> bool:
        return self.overlay.color_predictor.save_samples()

    def load_color_samples(self) -> bool:
        return self.overlay.color_predictor.load_latest_save_file()

    def calibrate(self, positions, selected_colors, n_frames=50) -> str:
        succ = True
        for _f in range(n_frames):
            frame_std = self.get_frame()
            if frame_std is None:
                return "No frame detected"
            succ = succ and self.overlay.calibrate(frame_std, positions, selected_colors)
        # succ = succ and self.save_color_samples()
        if succ:
            return "Calibration worked!"
        else:
            return "Could not write file (or other error)"

    def get_board_colors(self):# -> typing.Optional[dict[int, int]]:
        frame_std = self.get_frame()
        if frame_std is None:
            return None
        self.overlay.update_color_values(frame_std)
        return self.overlay.chess_board_values

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
