from tkinter import Frame, Listbox, Misc


class SettingsView(Frame):
    def __init__(self, master: Misc):
        super().__init__(master=master)

        self.columnconfigure(0, weight=10)
        self.columnconfigure(1, weight=90)

        self.profiles_frame = Frame(self)
        self.profiles_frame.grid(row=0, column=0, sticky="nsew")

        self.profiles_list = Listbox(self.profiles_frame, selectmode="single")
        self.profiles_list.pack(expand=True, fill="both")

    def load_profiles(self, items: list[str]):
        selected_value = self.selected_value()

        self.profiles_list.delete(first=0, last="end")
        for item in items:
            self.profiles_list.insert("end", item)

        if selected_value in items:
            idx = items.index(selected_value)
            self.profiles_list.selection_set(idx)

    def selected_value(self) -> str | None:
        curselection = self.profiles_list.curselection()
        return self.profiles_list.get(curselection[0]) if curselection else None
