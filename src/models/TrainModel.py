from multiprocessing import Event, Process, Queue
from typing import Any

from workers.TrainWorker import TrainWorker


class TrainModel:
    process: None | Process = None

    def __init__(self):
        self.event_stop = Event()
        self.queue_logs = Queue()

    def start(self, train_worker_args: dict[str, Any]):
        if self.process is not None and self.process.is_alive():
            raise RuntimeError(f"Worker (pid={self.process.pid}) is already running")

        self.event_stop.clear()

        self.process = Process(
            target=TrainWorker(**train_worker_args).run,
            args=(self.event_stop, self.queue_logs),
        )
        self.process.start()
