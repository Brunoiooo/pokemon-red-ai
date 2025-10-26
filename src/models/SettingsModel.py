import os


class SettingsModel:
    path = "profiles"

    def __init__(self):
        pass

    def get_profiles(self):
        os.makedirs(self.path, exist_ok=True)

        return [entry.name for entry in os.scandir(self.path) if entry.is_dir()]
