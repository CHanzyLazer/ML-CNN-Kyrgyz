# 绘图窗口，使用 tkinter 实现，需要能在窗口中调整笔刷大小和笔刷的圆润程度
import tkinter as tk
from time import time
from typing import Callable

import PIL.ImageTk, PIL.Image, PIL.ImageDraw
from math import sqrt, ceil
from image import *


# 获取两点中间的点，按照给定的 dr 进行插值，不包含两端点
def get_mid_points(x0: int, y0: int, x1: int, y1: int, dr:float=1.0)->list[tuple[int, int]]:
    point_list = []
    dx = x1 - x0
    dy = y1 - y0
    split_num = ceil(sqrt(dx*dx + dy*dy)/dr) # 只要距离大于 dr 就一定需要分点
    dx /= split_num
    dy /= split_num
    for i in range(1, split_num):
        x = x0 + round(dx * i)
        y = y0 + round(dy * i)
        if (x!=x0 or y!=y0) and (x!=x1 or y!=y1):
            point_list.append((x, y))
    return point_list

# 画笔类，直接在图像上绘制
class BrushBasic:
    def draw(self, drawer: PIL.ImageDraw, x: int, y: int): pass

class Pen(BrushBasic):
    def __init__(self, color=(0,0,0), width=10):
        self._color = color
        self._brushBitmap = PIL.Image.new('L', (width, width), 255)
        self.offset_x = -width//2-1
        self.offset_y = -width//2-1
    
    def draw(self, drawer: PIL.ImageDraw, x: int, y: int):
        drawer.bitmap((x+self.offset_x, y+self.offset_y), self._brushBitmap, fill=self._color)

class Eraser(Pen):
    def __init__(self, color=(255, 255, 255), width=20):
        super().__init__(color, width)
    

# 可绘图的画布类，实现了鼠标左键和右键绘图的方法，注意需要通过 set 修改左键右键的画笔类型
class DrawableLabel(tk.Label):
    def __init__(self, master, width:int=512, height:int=512, delay=50):
        self._delay = delay # 控制图像刷新频率
        super().__init__(master, width=width, height=height, padx=0, pady=0)
        # 指定图片
        self._image = PIL.Image.new('RGB', (width, height), (255, 255, 255))
        self._drawer = PIL.ImageDraw.Draw(self._image)
        # 绑定鼠标点击事件（绘画）
        self.bind('<Button-1>', self._mouseL_down)
        self.bind('<B1-Motion>', self._mouseL_move)
        self.bind('<ButtonRelease-1>', self._mouseL_up)
        self.bind('<Button-3>', self._mouseR_down)
        self.bind('<B3-Motion>', self._mouseR_move)
        self.bind('<ButtonRelease-3>', self._mouseR_up)
        # 设定左右键的画笔
        self._mouseL_brush = Pen()
        self._mouseR_brush = Eraser()
        # 一些状态值
        self._drawingL, self._drawingR = False, False
        self._updating = False
        self._p_x, self._p_y = 0, 0
        self.update_time = 0.0  # 记录更新花费的时间，在下次delay时减去
        # 绑定的更新列表
        self.call_in_refresh = []
    
    # 绑定更新图像时需要执行的操作，例如更新模型的预测值
    def bind_refresh(self, func: Callable):
        self.call_in_refresh.append(func)
        
    # 更新图像
    def refresh(self):
        begin = time()
        for func in self.call_in_refresh: func()
        self._imageTK = PIL.ImageTk.PhotoImage(self._image)
        self.configure(image=self._imageTK)
        self._updating = False
        self.update_time = (time() - begin)*1000
    
    # 获取 np 数组类型的图像
    def get_image(self)->np.ndarray:
        return pil_to_cv(self._image)
    
    # 需要更新时调用
    def _update(self):
        if self._updating: return
        self._updating = True # 防止重复更新
        self.after(max(round(self._delay-self.update_time), 0), self.refresh)
        
    # 清空图像
    def clear(self):
        self._image = PIL.Image.new('RGB', self._image.size, (255, 255, 255))
        self._drawer = PIL.ImageDraw.Draw(self._image)
        self._update()
    
    def _mouseL_down(self, event: tk.Event):
        self._drawingR = False
        self._drawingL = True
        self._mouseL_brush.draw(self._drawer, event.x, event.y)
        self._p_x = event.x
        self._p_y = event.y
        self._update()
    def _mouseL_move(self, event: tk.Event):
        if self._drawingL:
            for x, y in get_mid_points(self._p_x, self._p_y, event.x, event.y):
                self._mouseL_brush.draw(self._drawer, x, y)
            self._mouseL_brush.draw(self._drawer, event.x, event.y)
            self._p_x = event.x
            self._p_y = event.y
            self._update()
    def _mouseL_up(self, event: tk.Event):
        self._drawingL = False
    
    def _mouseR_down(self, event: tk.Event):
        self._drawingL = False
        self._drawingR = True
        self._mouseR_brush.draw(self._drawer, event.x, event.y)
        self._p_x = event.x
        self._p_y = event.y
        self._update()
    def _mouseR_move(self, event: tk.Event):
        if self._drawingR:
            for x, y in get_mid_points(self._p_x, self._p_y, event.x, event.y):
                self._mouseR_brush.draw(self._drawer, x, y)
            self._mouseR_brush.draw(self._drawer, event.x, event.y)
            self._p_x = event.x
            self._p_y = event.y
            self._update()
    def _mouseR_up(self, event: tk.Event):
        self._drawingR = False


# tkinter 提供的接口不太适合做可变大小的窗口，方便起见这里锁死
class DrawWindow:
    def __init__(self, title='Draw'):
        self.title = title
        
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry('800x535')
        self.window.resizable(width=False, height=False)

        # 画布
        self.draw_graph = DrawableLabel(self.window)
        self.draw_graph.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NS)
        self.draw_graph.bind_refresh(self._draw_graph_refreshed)
        
        # 画笔框
        self.brush_frame = tk.Frame(self.window, width=100, height=512, highlightthickness=1, highlightbackground='#909090')
        self.brush_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NW)
        # 清空图像的工具
        bin_img = PIL.Image.open('texture/bin.png').resize((48, 48), PIL.Image.Resampling.NEAREST)
        self.bin_imgTK = PIL.ImageTk.PhotoImage(bin_img)
        self.brush_clear = tk.Button(self.brush_frame, command=self.draw_graph.clear, highlightthickness=0, image=self.bin_imgTK)
        self.brush_clear.grid(row=0, column=0, padx=5, pady=5, sticky=tk.N)
        
        
    # 绘图发生了更新时调用
    def _draw_graph_refreshed(self):
        pass
        
    def get_draw(self)->np.ndarray:
        return self.draw_graph.get_image()
    
    def show(self):
        self.draw_graph.refresh()
        self.window.mainloop()
    
if __name__ == "__main__":
    DrawWindow().show()
    