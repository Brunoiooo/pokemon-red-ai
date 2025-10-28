from tkinter import messagebox
from typing import Any
from models.SettingsModel import SettingsModel
from views.SettingsView import SettingsView


class SettingsController:
    def __init__(self, view: SettingsView, model: SettingsModel):
        self.__view = view
        self.__model = model

        self.__view.string_var_create_profile.trace_add(
            "write", self.handle_string_var_create_profile
        )
        self.__view.button_create_profile.configure(
            command=self.handle_button_create_profile
        )
        self.__view.listbox_profiles.bind(
            "<<ListboxSelect>>", self.handle_listbox_profiles
        )
        self.__view.button_delete_profile.configure(
            command=self.handle_button_delete_profile
        )
        self.__view.boolean_var_settings_debug.trace_add(
            "write", self.handle_boolean_var_settings_debug
        )

        self.__refresh()

    def __refresh(self):
        self.__view.load_profiles(self.__model.profiles)
        self.__view.button_create_profile.configure(
            state=(
                "normal"
                if len(self.__view.string_var_create_profile.get())
                and self.__view.string_var_create_profile.get()
                not in self.__model.profiles
                else "disabled"
            )
        )
        self.__view.button_delete_profile.configure(
            state="normal" if self.__view.selected_value() else "disabled"
        )
        if self.__model.settings is not None:
            self.__view.load_settings(self.__model.settings)
            self.__view.frame_settings.grid(row=0, column=1, sticky="nsew")
        else:
            self.__view.frame_settings.grid_forget()

    def handle_string_var_create_profile(self, *args):
        self.__refresh()

    def handle_button_create_profile(self):
        try:
            self.__model.create_profile(self.__view.string_var_create_profile.get())
        except Exception as e:
            messagebox.showerror("handle_button_create_profile", e)
        finally:
            self.__refresh()

    def handle_listbox_profiles(self, *args):
        try:
            self.__model.profile = self.__view.selected_value()
        except Exception as e:
            messagebox.showerror("handle_listbox_profiles", e)
        finally:
            self.__refresh()

    def handle_button_delete_profile(self):
        try:
            profile = self.__view.selected_value()

            if not profile:
                raise ValueError("Profile has not selected.")

            self.__model.delete_profile(profile)
        except Exception as e:
            messagebox.showerror("handle_button_delete_profile", e)
        finally:
            self.__refresh()

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
