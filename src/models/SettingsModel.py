import json
import os
import shutil


class SettingsModel:
    path_base = "profiles"
    file_settings = "settings.json"

    @property
    def profiles(self):
        os.makedirs(self.path_base, exist_ok=True)

        return [entry.name for entry in os.scandir(self.path_base) if entry.is_dir()]

    def get_settings(self, profile: str):
        if profile not in self.profiles:
            raise ValueError(f"Profile {profile} does not exist.")

        with open(
            os.path.join(self.path_base, profile, self.file_settings),
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def set_settings(self, profile: str, settings: any):
        with open(
            os.path.join(self.path_base, profile, self.file_settings),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(settings, file, indent=4, ensure_ascii=False)

    def create_profile(self, profile: str):
        path = os.path.join(self.path_base, profile, self.file_settings)

        if os.path.isfile(path):
            raise ValueError(f"Profile {profile} already exists.")

        with open(path, "w", encoding="utf-8") as file:
            json.dump({}, file, indent=4, ensure_ascii=False)

    def delete_profile(self, profile: str):
        if profile not in self.profiles:
            raise ValueError("Profile does not exist.")

        shutil.rmtree(os.path.join(self.path_base, profile))
