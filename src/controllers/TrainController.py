import queue
from tkinter import messagebox
from models.TrainModel import TrainModel
from views.TrainView import TrainView


class TrainController:
    def __init__(self, view: TrainView, model: TrainModel):
        self.__view = view
        self.__model = model

        self.__view.button_start.config(command=self.start)
        self.__view.button_stop.config(command=self.stop)
        self.__view.button_auto_mode.config(command=self.start_auto_mode)

        self.__view.button_evaluate_greedy_best.config(
            command=self.start_evaluate_greedy_best
        )
        self.__view.button_evaluate_greedy_latest.config(
            command=self.start_evaluate_greedy_latest
        )

        self.__view.boolean_var_settings_debug.trace_add(
            "write", self.handle_boolean_var_settings_debug
        )

        self.__view.boolean_var_settings_evaluation_window.trace_add(
            "write", self.handle_boolean_var_settings_evaluation_window
        )

        self.__view.boolean_var_train_window.trace_add(
            "write", self.handle_boolean_var_train_window
        )

        self.__refresh()
        self.__run_logs()

    def __refresh(self):
        self.__view.button_start.configure(
            state=(
                "disabled"
                if self.__model.train_worker.event_start.is_set()
                else "normal"
            )
        )
        self.__view.button_stop.configure(
            state=(
                "disabled"
                if not self.__model.train_worker.event_start.is_set()
                else "normal"
            )
        )
        self.__view.button_evaluate_greedy_best.config(
            state=(
                "disabled"
                if self.__model.evaluate_process
                and self.__model.evaluate_process.is_alive()
                else "normal"
            )
        )
        self.__view.button_evaluate_greedy_latest.config(
            state=(
                "disabled"
                if self.__model.evaluate_process
                and self.__model.evaluate_process.is_alive()
                else "normal"
            )
        )
        self.__view.button_auto_mode.config(
            state=(
                "disabled"
                if self.__model.auto_mode_process
                and self.__model.auto_mode_process.is_alive()
                else "normal"
            )
        )

        self.__view.after(100, self.__refresh)

    def start(self):
        try:
            self.__model.start()
        except Exception as e:
            messagebox.showerror("start", e)

    def stop(self):
        try:
            self.__model.train_worker.event_start.clear()
        except Exception as e:
            messagebox.showerror("stop", e)

    def __run_logs(self):
        try:
            for _ in range(self.__view.int_var_logs_per_run.get()):
                self.__view.add_log(self.__model.train_worker.queue_logs.get_nowait())
        except queue.Empty:
            pass

        self.__view.after(
            self.__view.int_var_logs_per_run.get(),
            self.__run_logs,
        )

    def handle_boolean_var_settings_debug(self, *args):
        try:
            self.__model.train_worker.is_debug.value = (
                self.__view.boolean_var_settings_debug.get()
            )
        except Exception as e:
            messagebox.showerror("handle_boolean_var_settings_debug", e)

    def handle_boolean_var_settings_evaluation_window(self, *args):
        try:
            self.__model.train_worker.is_evaluation_window.value = (
                self.__view.boolean_var_settings_evaluation_window.get()
            )
        except Exception as e:
            messagebox.showerror("handle_boolean_var_settings_evaluation_window", e)

    def start_evaluate_greedy_best(self):
        try:
            self.__view.add_log("Starting evaluation with best model...")
            self.__view.button_evaluate_greedy_best.config(state="disabled")
            self.__view.button_evaluate_greedy_latest.config(state="disabled")
            self.__model.start_evaluation(best_model=True)

        except Exception as e:
            messagebox.showerror("start_evaluate_greedy_best", e)

    def start_evaluate_greedy_latest(self):
        try:
            self.__view.add_log("Starting evaluation with latest model...")
            self.__view.button_evaluate_greedy_best.config(state="disabled")
            self.__view.button_evaluate_greedy_latest.config(state="disabled")
            self.__model.start_evaluation(best_model=False)
        except Exception as e:
            messagebox.showerror("start_evaluate_greedy_latest", e)

    def start_auto_mode(self):
        try:
            self.__view.add_log("Starting Auto Mode...")
            self.__view.button_auto_mode.config(state="disabled")
            self.__model.start_auto_mode()
        except Exception as e:
            messagebox.showerror("start_auto_mode", e)

    def handle_boolean_var_train_window(self, *args):
        try:
            self.__model.train_worker.train_use_sdl.value = bool(
                self.__view.boolean_var_train_window.get()
            )
        except Exception as e:
            messagebox.showerror("handle_boolean_var_train_window", e)
