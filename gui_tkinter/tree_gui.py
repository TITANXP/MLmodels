import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from models.regression import CART_moddel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def re_draw(tolS, tolN):
    re_draw.f.clf()
    re_draw.a = re_draw.f.add_subplot(111)
    if chk_btn_var.get():
        tree = CART_moddel.create_tree(re_draw.raw_data)
        y_hat = CART_moddel.predict(tree, re_draw.test_data)
    else:
        tree = CART_moddel.create_tree(re_draw.raw_data)
        y_hat = CART_moddel.predict(tree, re_draw.test_data)
    re_draw.a.scatter(re_draw.raw_data[:,0], re_draw.raw_data[:,1], s=5)
    re_draw.a.plot(re_draw.test_data, y_hat)
    re_draw.canvas.show()

def get_inputs():
    """获取输入框的值"""
    try:
        tolN = int(tolN_entry.get())
    except:
        tolN = 10
        print('请输入tolN')
        tolN_entry.delete(0,tk.END)
        tolN_entry.insert(0, '10')
    try:
        tolS = float(tolS_entry.get())
    except:
        tolS = 1.0
        print('请输入tolS')
        tolS_entry.delete(0,tk.END)
        tolS_entry.insert(0, '10')
    return tolN, tolS

def draw_new_tree():
    tolN, tolS = get_inputs()
    re_draw(tolN, tolS)

if __name__ == '__main__':

    root = tk.Tk()
    tk.Label(root, text='plot place holder').grid(row=0, columnspan=3)
    tk.Label(root, text='tolN').grid(row=1, column=0)
    tolN_entry = tk.Entry(root)
    tolN_entry.grid(row=1, column=1)
    tolN_entry.insert(0,'10')
    tk.Label(root, text='tolS').grid(row=2, column=0)
    tolS_entry = tk.Entry(root)
    tolS_entry.grid(row=2, column=1)
    tolS_entry.insert(0,'1.0')

    tk.Button(root, text='重新画图', command=draw_new_tree).grid(row=1, column=2, rowspan=3)

    chk_btn_var = tk.IntVar()
    chk_btn = tk.Checkbutton(root, text='模型树', variable = chk_btn_var)
    chk_btn.grid(row=3, column=0, columnspan=2)
    re_draw.f = plt.Figure(figsize=(5,4), dpi=100)
    re_draw.canvas = FigureCanvasTkAgg(re_draw.f, root)
    re_draw.canvas.show()
    re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    re_draw.raw_data = CART_moddel.load_data()
    # re_draw.test_data = np.arange(min(re_draw.raw_data[:,0]), max(re_draw.raw_data[:,0]), 0.01)
    re_draw.test_data = CART_moddel.load_data()[:,0:-1]
    re_draw(0.02, 10)



    root.mainloop()