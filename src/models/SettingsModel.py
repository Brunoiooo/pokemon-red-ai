import json
import os
import shutil
from typing import Any


class SettingsModel:
    __PATH_BASE = "profiles"
    __FILE_SETTINGS = "settings.json"

    @property
    def profiles(self):
        os.makedirs(self.__PATH_BASE, exist_ok=True)

        return [entry.name for entry in os.scandir(self.__PATH_BASE) if entry.is_dir()]

    __profile: None | str = None

    @property
    def profile(self):
        return self.__profile

    @profile.setter
    def profile(self, profile: str | None):
        if profile and profile not in self.profiles:
            raise ValueError(f"Profile {profile} does not exist.")

        self.__profile = profile

    @property
    def settings(self) -> None | dict[str, Any]:
        if self.profile not in self.profiles:
            return None

        with open(
            os.path.join(self.__PATH_BASE, self.profile, self.__FILE_SETTINGS),
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @settings.setter
    def settings(self, settings: dict[str, Any]):
        if not self.profile:
            raise ValueError("Profile has not selected.")

        with open(
            os.path.join(self.__PATH_BASE, self.profile, self.__FILE_SETTINGS),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(settings, file, indent=4, ensure_ascii=False)

    def create_profile(self, profile: str):
        path = os.path.join(self.__PATH_BASE, profile, self.__FILE_SETTINGS)

        if os.path.isfile(path):
            raise ValueError(f"Profile {profile} already exists.")

        os.makedirs(os.path.join(self.__PATH_BASE, profile), exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump({}, file, indent=4, ensure_ascii=False)

    def delete_profile(self, profile: str):
        if profile not in self.profiles:
            raise ValueError("Profile does not exist.")

        if profile is self.profile:
            self.profile = None

        shutil.rmtree(os.path.join(self.__PATH_BASE, profile))
