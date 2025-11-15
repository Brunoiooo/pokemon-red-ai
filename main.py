import sys
import torch
import os
from multiprocessing import set_start_method

sys.path.append("src")

from Router import Router


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    set_start_method("spawn", force=True)

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

    Router().mainloop()


if __name__ == "__main__":
    main()
