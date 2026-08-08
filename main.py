#!/usr/bin/env python3
"""Deprecated Tk GUI entrypoint.

The TrainModel module is missing and the GUI is not on the PPO path.
Use the CLI instead:

  python cli.py train --workers 8
  python cli.py eval --model models/.../best/best_model.zip
"""
import sys


def main():
    print(
        "The Tk GUI (main.py) is deprecated / broken.\n"
        "Use the PPO CLI:\n"
        "  python cli.py train --workers 8 --stage stage_left_house\n"
        "  python cli.py eval --model models/<run>/best/best_model.zip\n"
        "\n"
        "Legacy DQN:\n"
        "  python cli.py train-dqn\n"
        "  python cli.py eval-dqn --model models/best.pth\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
