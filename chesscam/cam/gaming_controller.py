import pygame
import enum


class ControllerValueTypes(enum.Enum):
    KEY_DOWN = 1539
    KEY_UP = 1540
    JogWheel = 1536


class ControllerButtons(enum.Enum):
    A_BTN = 0
    B_BTN = 1
    X_BTN = 2
    Y_BTN = 3
    BACK_BTN = 6


class Game_Controller:
    def __init__(self):
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            # raise IOError("No Joystick connected")
            print("No Joystick connected")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Joystick detected ")

    def get_inputs(self):
        actions = []
        if self.joystick is not None:
            for event in pygame.event.get():
                button = None
                # The 0 button is the 'a' button, 1 is the 'b' button, 2 is the 'x' button, 3 is the 'y' button
                if event.type not in [v.value for v in ControllerValueTypes]:
                    continue
                if event.type in [ControllerValueTypes.KEY_DOWN.value, ControllerValueTypes.KEY_UP.value]:
                    button = ControllerButtons(event.button)
                actions.append((ControllerValueTypes(event.type), button, event))
        return actions