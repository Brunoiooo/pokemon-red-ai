import json
from multiprocessing import Event, Process, Queue, Value
import os
from typing import Any

from workers.TrainWorker import TrainWorker


class TrainModel:
    process: None | Process = None
    __FILE_SETTINGS = "settings.json"

    def __init__(self):
        self.event_stop = Event()
        self.event_stop.set()
        self.queue_logs = Queue()
        self.count = Value("i", 0)
        self.is_debug = Value("b", False)
        self.queue_evaluation_logs = Queue()
        self.is_evaluation_window = Value("b", False)

    @property
    def settings(self) -> dict[str, Any]:
        if not os.path.isfile(self.__FILE_SETTINGS):
            with open(
                self.__FILE_SETTINGS,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump({}, file, indent=4, ensure_ascii=False)

        with open(
            self.__FILE_SETTINGS,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @settings.setter
    def settings(self, settings: dict[str, Any]):
        with open(
            os.path.join(self.__FILE_SETTINGS),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(settings, file, indent=4, ensure_ascii=False)

    def start(self):
        if self.process is not None and self.process.is_alive():
            raise RuntimeError(f"Worker (pid={self.process.pid}) is already running")

        self.event_stop.clear()

        self.process = Process(
            target=TrainWorker().run,
            kwargs={
                "event_stop": self.event_stop,
                "queue_logs": self.queue_logs,
                "count": self.count,
                "is_debug": self.is_debug,
                "queue_evaluation_logs": self.queue_evaluation_logs,
                "is_evaluation_window": self.is_evaluation_window,
            },
            daemon=False,
        )
        self.process.start()
