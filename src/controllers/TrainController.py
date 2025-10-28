import queue
from tkinter import messagebox
from models.SettingsModel import SettingsModel
from models.TrainModel import TrainModel
from views.TrainView import TrainView
from workers import TrainWorker


class TrainController:
    def __init__(
        self, view: TrainView, model: TrainModel, settingsModel: SettingsModel
    ):
        self.__view = view
        self.__model = model
        self.__settingsModel = settingsModel

        self.__view.button_start.config(command=self.start)

        self.__refresh()

        self.__configure_button_start()

    def __refresh(self):
        pass

    def start(self):
        try:
            if self.__settingsModel.settings is None:
                raise ValueError("Profile has not selected.")

            self.__model.start(self.__settingsModel.settings)

            self.__run_logs()
        except Exception as e:
            messagebox.showerror("start", e)
        finally:
            self.__refresh()

    def __run_logs(self):
        try:
            for _ in range(self.__settingsModel.settings.get("logs_per_run", 100)):
                self.__view.add_log(self.__model.queue_logs.get_nowait())
        except queue.Empty:
            pass

        if self.__is_running:
            self.__view.after(
                self.__settingsModel.settings.get("logs_delay_ms", 50), self.__run_logs
            )

    def __configure_button_start(self):
        self.__view.button_start.configure(
            state=(
                "disabled"
                if not self.__model.process
                or not self.__model.process.is_alive()
                and self.__settingsModel.settings is None
                else "normal"
            )
        )

        self.__view.after(1000, self.__configure_button_start)
