from datetime import datetime
from tkinter import BooleanVar, IntVar, Misc
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk


class TrainView(ttk.Frame):
    PAD = 6

    def __init__(self, master: Misc):
        super().__init__(master)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=self.PAD, pady=(self.PAD, 2))
        toolbar.columnconfigure(10, weight=1)

        self.button_start = ttk.Button(toolbar, text="Start")
        self.button_stop = ttk.Button(toolbar, text="Stop")
        self.button_auto_mode = ttk.Button(toolbar, text="Auto Mode")
        self.button_start.grid(row=0, column=0, padx=(0, self.PAD))
        self.button_stop.grid(row=0, column=1)
        self.button_auto_mode.grid(row=0, column=2, padx=(self.PAD, 0))

        mid = ttk.Frame(self)
        mid.grid(row=1, column=0, sticky="ew", padx=self.PAD, pady=(2, 2))
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)

        settings = ttk.LabelFrame(mid, text="Ustawienia")
        settings.grid(row=0, column=0, sticky="nsew", padx=(0, self.PAD))
        for c in range(4):
            settings.columnconfigure(c, weight=1)

        ttk.Label(settings, text="Logs batch:").grid(
            row=0, column=0, sticky="e", padx=2, pady=2
        )
        self.int_var_logs_per_run = IntVar(value=50)
        self.spin_logs_per_run = ttk.Spinbox(
            settings, from_=1, to=9999, width=6, textvariable=self.int_var_logs_per_run
        )
        self.spin_logs_per_run.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(settings, text="Delay (ms):").grid(
            row=0, column=2, sticky="e", padx=2, pady=2
        )
        self.int_var_logs_delay_ms = IntVar(value=50)
        self.spin_logs_delay = ttk.Spinbox(
            settings, from_=0, to=9999, width=6, textvariable=self.int_var_logs_delay_ms
        )
        self.spin_logs_delay.grid(row=0, column=3, sticky="w", pady=2)

        self.boolean_var_settings_debug = BooleanVar()
        ttk.Checkbutton(
            settings, text="Debug", variable=self.boolean_var_settings_debug
        ).grid(row=1, column=0, sticky="w", padx=2, pady=2)

        self.boolean_var_settings_evaluation_window = BooleanVar()
        ttk.Checkbutton(
            settings,
            text="Debug Eval",
            variable=self.boolean_var_settings_evaluation_window,
        ).grid(row=1, column=1, sticky="w", padx=2, pady=2)

        ttk.Label(settings, text="Use SDL window:").grid(
            row=1, column=2, sticky="e", padx=2, pady=2
        )
        self.boolean_var_train_window = BooleanVar(value=False)
        ttk.Checkbutton(
            settings, text="SDL", variable=self.boolean_var_train_window
        ).grid(row=1, column=3, sticky="w", pady=2)

        actions = ttk.LabelFrame(mid, text="Ewaluacja")
        actions.grid(row=0, column=1, sticky="nsew")
        self.button_evaluate_greedy_latest = ttk.Button(
            actions, text="Evaluate Greedy Latest"
        )
        self.button_evaluate_greedy_best = ttk.Button(
            actions, text="Evaluate Greedy Best"
        )
        self.button_evaluate_greedy_latest.grid(
            row=0, column=0, sticky="ew", padx=4, pady=(4, 2)
        )
        self.button_evaluate_greedy_best.grid(
            row=1, column=0, sticky="ew", padx=4, pady=(0, 4)
        )

        logs_frame = ttk.LabelFrame(self, text="Logi")
        logs_frame.grid(
            row=2, column=0, sticky="nsew", padx=self.PAD, pady=(0, self.PAD)
        )
        logs_frame.rowconfigure(0, weight=1)
        logs_frame.columnconfigure(0, weight=1)

        self.scrolled_logs = ScrolledText(logs_frame, wrap="word", state="disabled")
        self.scrolled_logs.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    def add_log(self, log: str):
        self.scrolled_logs.configure(state="normal")
        self.scrolled_logs.insert(
            "end", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {log}\n"
        )
        self.scrolled_logs.delete(
            "1.0",
            f"{max(1, int(self.scrolled_logs.index('end').split('.')[0]) - 1000)}.0",
        )
        self.scrolled_logs.see("end")
        self.scrolled_logs.configure(state="disabled")
