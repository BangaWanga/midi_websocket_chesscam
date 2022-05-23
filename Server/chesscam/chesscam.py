enable_game_controller = False
import cv2
import camera
import overlay_New
from enum import Enum
if enable_game_controller:
    from game_controller import Game_Controller, ControllerValueTypes, ControllerButtons


class ControlKeys(Enum):
    MoveGridUp = 0
    MoveGridDown = 1
    MoveGridRight = 2
    MoveGridLeft = 3
    ZoomBigger = 4
    ZoomSmaller = 5
    ScrollDisplayOptions = 6
    TrainRed = 7
    TrainGreen = 8
    TrainBlue = 9



class ChessCam:
    def __init__(self):
        self.camera = camera.Camera()
        self.overlay = overlay_New.Overlay(self.camera.get_cam_resolution())    # handle scale and pos differently
        self.control_map = {
            97: ControlKeys.MoveGridUp,
            100: ControlKeys.MoveGridDown,
            119: ControlKeys.MoveGridRight,
            115: ControlKeys.MoveGridLeft,
            43: ControlKeys.ZoomBigger,
            45: ControlKeys.ZoomSmaller,
            57: ControlKeys.ScrollDisplayOptions,
            114: ControlKeys.TrainRed,
            103: ControlKeys.TrainGreen,
            98: ControlKeys.TrainBlue,
        }
        print("Chesscam init finished")
        self.full_screen = False
        self.display_size = 800, 600
        self.training_rgb = (False, False, False)
        self.fitting_rgb_range = (False, False, False)
        if enable_game_controller:
            self.game_controller = Game_Controller()

    def update(self):
        frame = self.camera.capture_frame_from_videostream()
        frame = self.overlay.draw_grid(frame)
#        if any(self.fitting_rgb_range):
#            self.overlay.add_sample(img, field_pos=(0, 0), color_label=self.fitting_rgb_range.index(True))
#            self.fitting_rgb_range = (False, False, False)
#        if any(self.training_rgb):
#            self.overlay.train_all_fields(img, self.training_rgb.index(True))
#            self.training_rgb = (False, False, False)
#            print("Trained")
        cv2.imshow('computer visions', frame)
        self.process_key_input()
        if enable_game_controller:
            self.process_controller_input()

    def process_controller_input(self):
        actions = self.game_controller.get_inputs()
        button_down = False
        button_up = False
        color_class = None

        for a in actions:
            if a[0] == ControllerValueTypes.JogWheel:
                self.overlay.move_cursor(a[2].axis, a[2].value)
            if a[0] == ControllerValueTypes.KEY_UP:
                button_up = True
            if a[0] == ControllerValueTypes.KEY_DOWN:
                button_down = True
            if a[1] == ControllerButtons.A_BTN:
                color_class = 0
            if a[1] == ControllerButtons.B_BTN:
                color_class = 1
            if a[1] == ControllerButtons.X_BTN:
                color_class = 2
            if a[1] == ControllerButtons.Y_BTN:
                color_class = 3
            if a[1] == ControllerButtons.BACK_BTN:
                if button_down:
                    self.overlay.color_predictor.save_samples() # save to config file
        if button_down and color_class is not None:
            self.overlay.select_field(color_class)

    def process_key_input(self):
        key = cv2.waitKey(1)
        if key == 113 or key == 27:
            self.quit()
        move_size = 10
        offset = self.overlay.offset
        scale = self.overlay.scale
        if key != -1:
            if key in self.control_map:
                action = self.control_map[key]
                if action == ControlKeys.MoveGridUp:
                    offset = (offset[0] - move_size, offset[1])
                if action == ControlKeys.MoveGridDown:
                    offset = (offset[0] + move_size, offset[1])
                if action == ControlKeys.MoveGridRight:
                    offset = (offset[0], offset[1] - move_size)
                if action == ControlKeys.MoveGridLeft:
                    offset = (offset[0], offset[1] + move_size)
                if action == ControlKeys.ZoomBigger:
                    scale += 0.01
                if action == ControlKeys.ZoomSmaller:
                    scale -= 0.01
                if action == ControlKeys.ScrollDisplayOptions:    # 9 on keyboard
                    pass
                if action == ControlKeys.TrainRed:    # R on keyboard
                    self.fitting_rgb_range = (True, False, False)
                if action == ControlKeys.TrainGreen:    # G on keyboard
                    self.fitting_rgb_range = (False, True, False)
                if action == ControlKeys.TrainBlue:    # B on keyboard
                    self.fitting_rgb_range = (False, False, True)
                else:
                    print("KEEY: ", key)
            self.overlay.change_drawing_options(offset, scale)

    def quit(self):
        # When everything done, release the capture
        self.camera.cam.release()
        cv2.destroyAllWindows()
        quit()


if __name__ == "__main__":
    cam = ChessCam()

    while True:
        cam.update()

with open("",
          "w") as outfile:
    pass
