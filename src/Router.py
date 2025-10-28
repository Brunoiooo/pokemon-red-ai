from tkinter import Tk
from tkinter.ttk import Notebook

from controllers.TrainController import TrainController
from models.TrainModel import TrainModel
from views.TrainView import TrainView


class Router(Tk):
    settings_model_profile: None | str = None

    def __init__(self):
        super().__init__()

        self.title("Pokemon Red AI")
        self.state("zoomed")

        self.__notebook = Notebook(self)
        self.__notebook.pack(expand=True, fill="both")

        self.__train_view = TrainView(self.__notebook)
        self.__train_model = TrainModel()
        TrainController(
            model=self.__train_model,
            view=self.__train_view,
        )
        self.__notebook.add(self.__train_view, text="Train")
