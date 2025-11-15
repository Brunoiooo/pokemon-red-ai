import queue
import sys
from time import sleep
import torch
import os
from multiprocessing import set_start_method

sys.path.append("src")
import torch

print(torch.cuda.is_available())

from models.TrainModel import TrainModel


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision("high")

    set_start_method("spawn", force=True)

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

    trainModel = TrainModel()

    trainModel.start()
    trainModel.is_evaluation_window.value = True
    try:
        while True:
            try:
                print(trainModel.queue_logs.get_nowait())
            except queue.Empty:
                sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        trainModel.event_stop.set()


if __name__ == "__main__":
    main()
