import os
import shutil
from tkinter import Button, Entry, Event, Misc, StringVar, Tk, ttk, Listbox
from typing import Callable


class Window(Tk):
    def __init__(self):
        super().__init__()

        self.title("Pokemon Red AI")
        self.state("zoomed")

        self.notebook = Notebook(self, onSelect=self.render)
        self.notebook.pack(expand=True, fill="both")

        self.render()

    def render(self):
        self.notebook.render()


class Notebook(ttk.Notebook):
    def __init__(self, master: Misc, onSelect: Callable[[], None]):
        super().__init__(master)

        self.settingsFrame = SettingsFrame(self, onSelect=onSelect)
        self.add(self.settingsFrame, text="Settings")

    def render(self):
        self.settingsFrame.render()


class SettingsFrame(ttk.Frame):
    def __init__(self, master: Misc, onSelect: Callable[[], None]):
        super().__init__(master)

        self.columnconfigure(0, weight=10)
        self.columnconfigure(1, weight=90)

        self.profilesFrame = ProfilesFrame(self, onSelect=onSelect)
        self.profilesFrame.grid(column=0, row=0, sticky="nsew")

    def render(self):
        self.profilesFrame.render()


class ProfilesFrame(ttk.Frame):
    def __init__(self, master, onSelect: Callable[[], None]):
        super().__init__(master)

        self.stringVar = StringVar()
        self.stringVar.trace_add("write", self.write)

        self.entry = Entry(self, textvariable=self.stringVar)
        self.entry.pack(expand=True, fill="both")

        self.insertButton = Button(self, command=self.insert, text="New Profile")
        self.insertButton.pack(expand=True, fill="both")

        self.listbox = ProfilesListbox(self, onSelect=onSelect)
        self.listbox.pack(expand=True, fill="both")

        self.deleteButton = Button(self, command=self.delete, text="Delete Profile")
        self.deleteButton.pack(expand=True, fill="both")

    def render(self):
        self.insertButton.config(
            state="normal" if 1 <= len(self.entry.get()) else "disabled"
        )
        self.deleteButton.config(
            state="normal" if self.listbox.profile is not None else "disabled"
        )
        self.listbox.render()

    def write(self, a: str, b: str, c: str):
        self.render()

    def insert(self):
        profile = self.entry.get()

        if len(profile) <= 0:
            raise ValueError("len(profile) <= 0")

        os.makedirs(os.path.join(self.listbox.path, profile), exist_ok=True)

        self.render()

    def delete(self):
        if self.listbox.profile is None:
            raise ValueError("self.listbox.profile is None")

        shutil.rmtree(os.path.join(self.listbox.path, self.listbox.profile))

        self.render()


class ProfilesListbox(Listbox):
    path = "profiles"

    profile: str | None = None

    def __init__(self, master: Misc, onSelect: Callable[[], None]):
        super().__init__(master, selectmode="single")

        self.onSelect = onSelect

        self.bind("<<ListboxSelect>>", self._onSelect)

    def render(self):
        self.delete(first=0, last="end")

        os.makedirs(self.path, exist_ok=True)

        for entry in os.scandir(self.path):
            if entry.is_dir():
                self.insert("end", entry.name)

        self.selection_clear(0, "end")
        self.profile = None

    def _onSelect(self, event: Event):
        widget: Listbox = event.widget

        curselection = widget.curselection()

        self.profile = widget.get(curselection[0]) if 1 <= len(curselection) else None

        self.onSelect()
