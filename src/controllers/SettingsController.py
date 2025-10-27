from tkinter import messagebox
from models.SettingsModel import SettingsModel
from views.SettingsView import SettingsView


class SettingsController:
    settings: any | None = None

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
            state="normal" if self.model.profile else "disabled"
        )

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
            profile = self.view.selected_value()

            self.settings = self.model.get_settings(profile) if profile else None
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
