#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/10/24 

# 查看编码后的图

import tkinter as tk
import tkinter.ttk as ttk
from argparse import ArgumentParser
from traceback import print_exc

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from utils import *

FIG_SIZE = (10, 10)
MOUSE_WHEEL_SPD = 120


class App:

  def __init__(self, args):
    self.dataset = get_dataset(args)
    self.setup_gui()
    self.sc_idx.config(to=len(self.dataset)-1)
    self.redraw()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    wnd.title('Image Viewer')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd
    self.wnd.bind_all('<MouseWheel>', lambda evt: self.redraw(evt))

    # top: img-idx
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
    if True:
      self.var_idx = tk.IntVar(frm1, value=0)

      frm11 = ttk.Label(frm1)
      frm11.pack(expand=tk.YES, fill=tk.X)
      if True:
        ttk.Label(frm11, text='Image Index').pack(side=tk.LEFT)
        sc = tk.Scale(frm11, command=lambda _: self.redraw(), variable=self.var_idx, orient=tk.HORIZONTAL, from_=0, to=1, resolution=1, tickinterval=100)
        sc.pack(expand=tk.YES, fill=tk.X)
        self.sc_idx = sc

    # bottom: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      fig = plt.gcf()
      axs = [
        plt.subplot(221),
        plt.subplot(222),
        plt.subplot(212),
      ]
      fig.tight_layout()
      fig.set_size_inches(FIG_SIZE)
      cvs = FigureCanvasTkAgg(fig, frm2)
      toolbar = NavigationToolbar2Tk(cvs)
      toolbar.update()
      cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
      self.fig, self.axs, self.cvs = fig, axs, cvs

  def redraw(self, evt=None):
    if evt is None:
      idx = self.var_idx.get()
    else:
      idx = self.var_idx.get()
      idx = max(0, min(len(self.dataset)-1, idx - int(evt.delta / MOUSE_WHEEL_SPD)))
      self.var_idx.set(idx)
    fid, im_x, vec_x, im_z, vec_z = self.dataset[idx]

    # 布局: https://blog.csdn.net/sunshihua12829/article/details/52786144
    for ax in self.axs: ax.cla()
    self.axs[0].imshow(im_x) ; self.axs[0].set_title('x')
    self.axs[1].imshow(im_z) ; self.axs[1].set_title('z')
    self.axs[2].plot(vec_x, 'b', label='vec_x')
    self.axs[2].plot(vec_z, 'r', label='vec_z')
    self.axs[2].legend()
    self.fig.suptitle(f'[img-{idx}] Fid: {fid}')
    self.fig.tight_layout()
    self.cvs.draw()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-F', '--fp', default='./data/test_dataset.pkl', help='path to encode test_dataset.pkl')
  args = parser.parse_args()

  App(args)
