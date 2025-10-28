from tkinter import Button, Frame, Misc


class TrainView(Frame):
    def __init__(self, master: Misc):
        super().__init__(master=master)

        self.button_start = Button(self)
        self.button_start.grid(column=0, row=0)
