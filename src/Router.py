from tkinter import Tk
from tkinter.ttk import Notebook

from controllers.TrainController import TrainController
from controllers.SettingsController import SettingsController
from models.TrainModel import TrainModel
from models.SettingsModel import SettingsModel
from views.TrainView import TrainView
from views.SettingsView import SettingsView


class Router(Tk):
    def __init__(self):
        super().__init__()

        self.title("Pokemon Red AI")
        self.state("zoomed")

        notebook = Notebook(self)
        notebook.pack(expand=True, fill="both")

        settings_view = SettingsView(notebook)
        settingsModel = SettingsModel()
        SettingsController(model=settingsModel, view=settings_view)
        notebook.add(settings_view, text="Settings")

        train_view = TrainView(notebook)
        TrainController(
            model=TrainModel(), view=train_view, settingsModel=settingsModel
        )
        notebook.add(train_view, text="Train")
