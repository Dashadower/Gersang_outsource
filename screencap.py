# -*- coding:utf-8 -*-
import win32gui, time
from PIL import ImageGrab

gersang_window_title = "Gersang"

class MapleScreenCapturer:
    """Container for capturing MS screen"""
    def __init__(self):
        self.hwnd = None

    def ms_get_screen_hwnd(self):
        window_hwnd = win32gui.FindWindow(0, gersang_window_title)
        if not window_hwnd:
            return 0
        else:
            return window_hwnd

    def ms_get_screen_rect(self, hwnd):
        return win32gui.GetWindowRect(hwnd)

    def capture(self, set_focus=True, hwnd=None, rect=None):

        if hwnd:
            self.hwnd = hwnd
        if not hwnd:
            self.hwnd = self.ms_get_screen_hwnd()
        if not rect:
            rect = self.ms_get_screen_rect(self.hwnd)
        if set_focus:
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.1)
        img = ImageGrab.grab(rect)
        if img:
            return img
        else:
            return 0

