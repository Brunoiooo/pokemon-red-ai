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

        self.__view.boolean_var_settings_debug.trace_add(
            "write", self.handle_boolean_var_settings_debug
        )

        self.__refresh()
        self.__run_logs()

    def __refresh(self):
        self.__view.button_start.configure(
            state=(
                "disabled"
                if self.__model.process
                and self.__model.process.is_alive()
                or not self.__model.event_stop.is_set()
                else "normal"
            )
        )
        self.__view.button_stop.configure(
            state=(
                "disabled"
                if self.__model.event_stop.is_set()
                or not self.__model.process
                or not self.__model.process.is_alive()
                else "normal"
            )
        )

        self.__view.after(1000, self.__refresh)

    def start(self):
        try:
            self.__model.start()

            self.__run_logs()
        except Exception as e:
            messagebox.showerror("start", e)
        finally:
            self.__refresh()

    def stop(self):
        try:
            self.__model.event_stop.set()
        except Exception as e:
            messagebox.showerror("stop", e)
        finally:
            self.__refresh()

    def __run_logs(self):
        try:
            for _ in range(self.__view.int_var_logs_per_run.get()):
                self.__view.add_log(self.__model.queue_logs.get_nowait())
        except queue.Empty:
            pass

        self.__view.after(
            self.__view.int_var_logs_per_run.get(),
            self.__run_logs,
        )

    def handle_boolean_var_settings_debug(self, *args):
        try:
            if self.__model.settings is None:
                raise ValueError("Profile has not selected.")

            settings = self.__model.settings
            settings["is_debug"] = self.__view.boolean_var_settings_debug.get()
            self.__model.settings = settings
        except Exception as e:
            messagebox.showerror("handle_boolean_var_settings_debug", e)
        finally:
            self.__refresh()
