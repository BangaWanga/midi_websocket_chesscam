import cv2
from Server.chesscam.camera import Camera
from Server.chesscam.overlay_New import Overlay
from enum import Enum
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
        self.camera = Camera()
        self.overlay = Overlay(self.camera.get_cam_resolution())    # handle scale and pos differently
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
        self.process_controller_input()

    def process_controller_input(self):
        actions = self.game_controller.get_inputs()
        button_down = False
        button_up = False
        color_class = None

        for a in actions:
            match a[0]:
                case ControllerValueTypes.JogWheel:
                    self.overlay.move_cursor(a[2].axis, a[2].value)
                case ControllerValueTypes.KEY_UP:
                    button_up = True
                case ControllerValueTypes.KEY_DOWN:
                    button_down = True
            match a[1]:
                case ControllerButtons.A_BTN:
                    color_class = 0
                case ControllerButtons.B_BTN:
                    color_class = 1
                case ControllerButtons.X_BTN:
                    color_class = 2
                case ControllerButtons.Y_BTN:
                    color_class = 3
                case ControllerButtons.BACK_BTN:
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
                match action:
                    case ControlKeys.MoveGridUp:
                        offset = (offset[0] - move_size, offset[1])
                    case ControlKeys.MoveGridDown:
                        offset = (offset[0] + move_size, offset[1])
                    case ControlKeys.MoveGridRight:
                        offset = (offset[0], offset[1] - move_size)
                    case ControlKeys.MoveGridLeft:
                        offset = (offset[0], offset[1] + move_size)
                    case ControlKeys.ZoomBigger:
                        scale += 0.01
                    case ControlKeys.ZoomSmaller:
                        scale -= 0.01
                    case ControlKeys.ScrollDisplayOptions:    # 9 on keyboard
                        self.overlay.scroll_display_option()
                    case ControlKeys.TrainRed:    # R on keyboard
                        self.fitting_rgb_range = (True, False, False)
                    case ControlKeys.TrainGreen:    # G on keyboard
                        self.fitting_rgb_range = (False, True, False)
                    case ControlKeys.TrainBlue:    # B on keyboard
                        self.fitting_rgb_range = (False, False, True)
                    case _:
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
