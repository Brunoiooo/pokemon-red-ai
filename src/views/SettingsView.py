from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    Frame,
    Listbox,
    Misc,
    StringVar,
)
from typing import Any


class SettingsView(Frame):
    def __init__(self, master: Misc):
        super().__init__(master=master)

        self.columnconfigure(0, weight=10)
        self.columnconfigure(1, weight=90)

        self.frame_profiles = Frame(self)
        self.frame_profiles.grid(row=0, column=0, sticky="nsew")

        self.string_var_create_profile = StringVar()

        self.entry_create_profile = Entry(
            self.frame_profiles, textvariable=self.string_var_create_profile
        )
        self.entry_create_profile.pack(expand=True, fill="both")

        self.button_create_profile = Button(self.frame_profiles, text="Create")
        self.button_create_profile.pack(expand=True, fill="both")

        self.listbox_profiles = Listbox(self.frame_profiles, selectmode="single")
        self.listbox_profiles.pack(expand=True, fill="both")

        self.button_delete_profile = Button(self.frame_profiles, text="Delete")
        self.button_delete_profile.pack(expand=True, fill="both")

        self.frame_settings = Frame(self)

        self.boolean_var_settings_debug = BooleanVar()

        self.checkbutton_settings_debug = Checkbutton(
            self.frame_settings, text="Debug", variable=self.boolean_var_settings_debug
        )
        self.checkbutton_settings_debug.grid(row=0, column=0)

    def load_profiles(self, items: list[str]):
        selected_value = self.selected_value()

        self.listbox_profiles.delete(first=0, last="end")
        for item in items:
            self.listbox_profiles.insert("end", item)

        if selected_value in items:
            idx = items.index(selected_value)
            self.listbox_profiles.selection_set(idx)

    def selected_value(self) -> str | None:
        curselection = self.listbox_profiles.curselection()
        return self.listbox_profiles.get(curselection[0]) if curselection else None

    def load_settings(self, settings: Any):
        self.boolean_var_settings_debug.set(bool(settings.get("debug")))
