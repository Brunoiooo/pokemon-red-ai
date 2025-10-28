from tkinter import messagebox
from typing import Any
from models.SettingsModel import SettingsModel
from views.SettingsView import SettingsView


class SettingsController:
    def __init__(self, view: SettingsView, model: SettingsModel):
        self.view = view
        self.model = model

        self.view.string_var_create_profile.trace_add(
            "write", self.handle_string_var_create_profile
        )
        self.view.button_create_profile.configure(
            command=self.handle_button_create_profile
        )
        self.view.listbox_profiles.bind(
            "<<ListboxSelect>>", self.handle_listbox_profiles
        )
        self.view.button_delete_profile.configure(
            command=self.handle_button_delete_profile
        )
        self.view.boolean_var_settings_debug.trace_add(
            "write", self.handle_boolean_var_settings_debug
        )

        self.refresh()

    def refresh(self):
        self.view.load_profiles(self.model.profiles)
        self.view.button_create_profile.configure(
            state=(
                "normal"
                if len(self.view.string_var_create_profile.get())
                and self.view.string_var_create_profile.get() not in self.model.profiles
                else "disabled"
            )
        )
        self.view.button_delete_profile.configure(
            state="normal" if self.view.selected_value() else "disabled"
        )
        if self.model.settings is not None:
            self.view.load_settings(self.model.settings)
            self.view.frame_settings.grid(row=0, column=1, sticky="nsew")
        else:
            self.view.frame_settings.grid_forget()

    def handle_string_var_create_profile(self, *args):
        self.refresh()

    def handle_button_create_profile(self):
        try:
            self.model.create_profile(self.view.string_var_create_profile.get())
        except Exception as e:
            messagebox.showerror("handle_button_create_profile", e)
        finally:
            self.refresh()

    def handle_listbox_profiles(self, *args):
        try:
            self.model.profile = self.view.selected_value()
        except Exception as e:
            messagebox.showerror("handle_listbox_profiles", e)
        finally:
            self.refresh()

    def handle_button_delete_profile(self):
        try:
            profile = self.view.selected_value()

            if not profile:
                raise ValueError("Profile has not selected.")

            self.model.delete_profile(profile)
        except Exception as e:
            messagebox.showerror("handle_button_delete_profile", e)
        finally:
            self.refresh()

    def handle_boolean_var_settings_debug(self, *args):
        try:
            if self.model.settings is None:
                raise ValueError("Profile has not selected.")

            settings = self.model.settings
            settings["is_debug"] = self.view.boolean_var_settings_debug.get()
            self.model.settings = settings
        except Exception as e:
            messagebox.showerror("handle_boolean_var_settings_debug", e)
        finally:
            self.refresh()
