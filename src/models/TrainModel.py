import json
from multiprocessing import Event, Process, Queue, Value
import os
from typing import Any

from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import get_model
from workers.TrainWorker import TrainWorker


class TrainModel:
    process: None | Process = None
    __FILE_SETTINGS = "settings.json"
    evaluate_process: None | Process = None
    auto_mode_process: None | Process = None

    def __init__(self):
        self.event_stop = Event()
        self.event_stop.set()
        self.queue_logs = Queue()
        self.count = Value("i", 0)
        self.is_debug = Value("b", False)
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
                "is_evaluation_window": self.is_evaluation_window,
            },
            daemon=False,
        )
        self.process.start()

    def start_evaluation(self, best_model: bool):
        if self.evaluate_process is not None and self.evaluate_process.is_alive():
            raise RuntimeError(
                f"Evaluation Worker (pid={self.evaluate_process.pid}) is already running"
            )

        model = get_model("cpu", "best" if best_model else "latest")
        model.eval()

        self.evaluate_process = Process(
            target=Emulator().evaluate_greedy,
            kwargs={
                "model": model,
                "evaluate_greedy_times": 1,
                "queue_logs": self.queue_logs,
                "is_debug": self.is_debug,
                "is_evaluation_window": self.is_evaluation_window,
            },
        )

        self.evaluate_process.start()

    def start_auto_mode(self):
        if self.auto_mode_process is not None and self.auto_mode_process.is_alive():
            raise RuntimeError(
                f"Auto Mode Worker (pid={self.auto_mode_process.pid}) is already running"
            )

        self.auto_mode_process = Process(
            target=Emulator().auto_mode,
            kwargs={"queue_logs": self.queue_logs},
        )

        self.auto_mode_process.start()
