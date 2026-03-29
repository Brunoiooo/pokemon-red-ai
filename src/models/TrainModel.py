import json
from multiprocessing import Manager, Process
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
    dots: list[tuple[float, float]] = []

    def __init__(self):
        self.train_worker = TrainWorker()

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

        self.train_worker.run()

    def start_evaluation(self, best_model: bool):
        if self.evaluate_process is not None and self.evaluate_process.is_alive():
            raise RuntimeError(
                f"Evaluation Worker (pid={self.evaluate_process.pid}) is already running"
            )

        self.evaluate_process = Process(
            target=Emulator(files_lock=self.train_worker.files_lock).evaluate_greedy,
            kwargs={
                "model_state_dict": get_model(
                    "cpu",
                    files_lock=self.train_worker.files_lock,
                    name="best" if best_model else "latest",
                ).state_dict(),
                "queue_logs": self.train_worker.queue_logs,
                "is_debug": self.train_worker.is_debug.value,
                "is_evaluation_window": self.train_worker.is_evaluation_window,
            },
        )

        self.evaluate_process.start()

    def start_auto_mode(self):
        if self.auto_mode_process is not None and self.auto_mode_process.is_alive():
            raise RuntimeError(
                f"Auto Mode Worker (pid={self.auto_mode_process.pid}) is already running"
            )

        self.auto_mode_process = Process(
            target=Emulator(files_lock=self.train_worker.files_lock).auto_mode,
            kwargs={"queue_logs": self.train_worker.queue_logs},
        )

        self.auto_mode_process.start()
