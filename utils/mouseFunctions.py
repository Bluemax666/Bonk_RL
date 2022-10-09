# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:11:01 2022

@author: Maxime
"""
import win32api
import win32con
import time

def getMousePos():
    return win32api.GetCursorPos()

def setMousePos(x, y):
    win32api.SetCursorPos((x,y))

def leftClick(x, y, duration=0.1):
    try:
        win32api.SetCursorPos((x,y))
        time.sleep(0.016)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        time.sleep(duration)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    except Exception as e:
        print(e)
    
def rightClick(x, y, duration=0.1):
    try:
        win32api.SetCursorPos((x,y))
        time.sleep(0.016)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
        time.sleep(duration)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
    except Exception as e:
        print(e)
    
    
#https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes

def isLeftMousePressed():
    return True if win32api.GetKeyState(0x01) < 0 else False

def isRightMousePressed():
    return True if win32api.GetKeyState(0x02) < 0 else False


if __name__ == '__main__':
    time.sleep(2)
    while not isLeftMousePressed():
        pass
    print(getMousePos())
    while isLeftMousePressed():
        pass
    print(getMousePos())
    
    