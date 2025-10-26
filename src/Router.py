from tkinter import Tk
from tkinter.ttk import Notebook

from controllers.SettingsController import SettingsController
from models.SettingsModel import SettingsModel
from views.SettingsView import SettingsView


class Router(Tk):
    def __init__(self):
        super().__init__()

        self.title("Pokemon Red AI")
        self.state("zoomed")

        self.notebook = Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        self.settings_model = SettingsModel()
        self.settings_view = SettingsView(self.notebook)
        self.settings_controller = SettingsController(
            model=self.settings_model, view=self.settings_view
        )
        self.notebook.add(self.settings_view, text="Settings")
