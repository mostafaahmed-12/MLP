import tkinter as tk
from tkinter import *
import Neural_Network as nn


def return_bias():
    if bias.get() == 1:
        return True
    else:
        return False


def return_actvation():
    if sig.get() == 1:
        return "sigmoid"
    elif tang.get() == 1:
        return "tanh"


def _run():
    text = entry2.get()
    m = text.split(",")
    neurons = []
    for i in m:
        neurons.append(int(i))

    nn.run(int(entry1.get()),float(entry3.get()),neurons,int(entry4.get()),return_actvation(),return_bias())



top = tk.Tk()

label1 = Label(top, text="Enter number of hidden layers:")
label1.place(x=350, y=150)
label1.pack()
entry1 = Entry(top, width=30)
entry1.place(x=350, y=170)
entry1.pack()

label2 = Label(top, text="Enter number of neurons in each hidden layer:")
label2.place(x=350, y=210)
label2.pack()
entry2 = Entry(top, width=30)
entry2.place(x=350, y=230)
entry2.pack()

label3 = Label(top, text="Enter learning rate:")
label3.place(x=350, y=270)
label3.pack()
entry3 = Entry(top, width=30)
entry3.place(x=350, y=290)
entry3.pack()

label4 = Label(top, text="Enter number of epochs:")
label4.place(x=350, y=330)
label4.pack()

entry4 = Entry(top, width=30)
entry4.place(x=350, y=350)
entry4.pack()

bias = tk.IntVar()
tk.Label(top, text="bias : ").pack()
tk.Checkbutton(top, text="bias", variable=bias, command=return_bias).pack()

tk.Label(top, text=" choose the activation function ").pack()
sig = tk.BooleanVar()
tang = tk.BooleanVar()
tk.Checkbutton(top, text="sigmoid", variable=sig, onvalue=1, offvalue=0, ).pack()
tk.Checkbutton(top, text="tanh", variable=tang, onvalue=1, offvalue=0, ).pack()

button_test = tk.Button(top, text="Run", command=_run)
button_test.place(x=700, y=500)
button_test.pack()

exit_button = tk.Button(top, text="Exit", command=top.destroy)
exit_button.pack()

top.mainloop()
