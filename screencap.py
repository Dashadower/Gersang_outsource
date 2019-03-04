# -*- coding:utf-8 -*-
import win32gui, time, ctypes
from PIL import ImageGrab

gersang_window_title = "Gersang"

class ScreenCapturer:
    """Container for capturing specific screen"""
    def __init__(self):
        self.hwnd = None

    def ms_get_screen_hwnd(self):
        window_hwnd = win32gui.FindWindow(0, gersang_window_title)
        if not window_hwnd:
            return 0
        else:
            return window_hwnd

    def ms_get_screen_rect(self, hwnd):
        """
        Added compatibility code from
        https://stackoverflow.com/questions/51786794/using-imagegrab-with-bbox-from-pywin32s-getwindowrect
        :param hwnd: window handle from self.ms_get_screen_hwnd
        :return: window rect (x1, y1, x2, y2) of MS rect.
        """
        try:
            f = ctypes.windll.dwmapi.DwmGetWindowAttribute
        except WindowsError:
            f = None
        if f:  # Vista & 7 stuff
            rect = ctypes.wintypes.RECT()
            DWMWA_EXTENDED_FRAME_BOUNDS = 9
            f(ctypes.wintypes.HWND(self.ms_get_screen_hwnd()),
              ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
              ctypes.byref(rect),
              ctypes.sizeof(rect)
              )
            size = (rect.left, rect.top, rect.right, rect.bottom)
        else:
            if not hwnd:
                hwnd = self.ms_get_screen_hwnd()
            size = win32gui.GetWindowRect(hwnd)
        return size  # returns x1, y1, x2, y2

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

