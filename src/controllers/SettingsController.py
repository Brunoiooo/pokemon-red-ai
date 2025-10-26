from models.SettingsModel import SettingsModel
from views.SettingsView import SettingsView


class SettingsController:
    def __init__(self, view: SettingsView, model: SettingsModel):
        self.view = view
        self.model = model

        self.refresh()

    def refresh(self):
        self.view.load_profiles(self.model.get_profiles())
