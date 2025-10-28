from tkinter import BooleanVar, Button, Checkbutton, Entry, Frame, IntVar, Label, Misc
from tkinter.scrolledtext import ScrolledText


class TrainView(Frame):
    def __init__(self, master: Misc):
        super().__init__(master=master)

        self.grid_rowconfigure(1, weight=1)
        for c in range(7):
            self.grid_columnconfigure(c, weight=1)

        self.button_start = Button(self, text="Start")
        self.button_start.grid(row=0, column=0, padx=6, pady=6, sticky="w")

        self.button_stop = Button(self, text="Stop")
        self.button_stop.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        Label(self, text="Logs batch").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.int_var_logs_per_run = IntVar(value=50)
        self.entry_logs_per_run = Entry(
            self, textvariable=self.int_var_logs_per_run, width=6
        )
        self.entry_logs_per_run.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        Label(self, text="Logs delay ms").grid(
            row=0, column=4, padx=5, pady=5, sticky="e"
        )
        self.int_var_logs_delay_ms = IntVar(value=50)
        self.entry_logs_delay_ms = Entry(
            self, textvariable=self.int_var_logs_delay_ms, width=6
        )
        self.entry_logs_delay_ms.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        self.boolean_var_settings_debug = BooleanVar()
        self.checkbutton_settings_debug = Checkbutton(
            self, text="Debug", variable=self.boolean_var_settings_debug
        )
        self.checkbutton_settings_debug.grid(
            row=0, column=6, padx=6, pady=6, sticky="e"
        )

        self.scrolled_logs = ScrolledText(self, wrap="word", state="disabled")
        self.scrolled_logs.grid(
            row=1, column=0, columnspan=7, sticky="nsew", padx=6, pady=(0, 6)
        )

    def add_log(self, log: str):
        self.scrolled_logs.configure(state="normal")
        self.scrolled_logs.insert("end", f"{log}\n")
        self.scrolled_logs.configure(state="disabled")
