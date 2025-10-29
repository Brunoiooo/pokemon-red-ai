import json
from multiprocessing import Event, Process, Queue
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

    def start(self, train_worker_args: dict[str, Any]):
        if self.process is not None and self.process.is_alive():
            raise RuntimeError(f"Worker (pid={self.process.pid}) is already running")

        self.event_stop.clear()

        self.process = Process(
            target=TrainWorker(**train_worker_args).run,
            args=(self.event_stop, self.queue_logs),
            daemon=True,
        )
        self.process.start()
