from multiprocessing import Queue
from multiprocessing.synchronize import Event


class TrainWorker:
    def run(self, event_stop: Event, queue_logs: Queue):
        pass
