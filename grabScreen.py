import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api


def grabScreen(region=None):
    
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, wind):
        # find the handle for the window we want to capture
        # self.hwnd = win32gui.FindWindow(None, window_name)
        # if not self.hwnd:
        #     raise Exception('Window not found: {}'.format(window_name))
            
        self.hwnd = wind

        # # get the window size
        # window_rect = win32gui.GetWindowRect(self.hwnd)
        # self.w = window_rect[2] - window_rect[0]
        # self.h = window_rect[3] - window_rect[1]

        # # account for the window border and titlebar and cut them off
        # border_pixels = 8
        # titlebar_pixels = 30
        # self.w = self.w - (border_pixels * 2)
        # self.h = self.h - titlebar_pixels - border_pixels
        # self.cropped_x = border_pixels
        # self.cropped_y = titlebar_pixels

        # # set the cropped coordinates offset so we can translate screenshot
        # # images into actual screen positions
        # self.offset_x = window_rect[0] + self.cropped_x
        # self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        #dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        dataBitMap.CreateCompatibleBitmap(dcObj, 600, 600)
        """rien"""
        cDC.SelectObject(dataBitMap)
        #cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)
        cDC.BitBlt((20, 20), (600, 600), dcObj, (0, 0), win32con.SRCCOPY)
        
        """capture marche pas"""
        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        #img.shape = (self.h, self.w, 4)
        img.shape = (600, 600, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img
