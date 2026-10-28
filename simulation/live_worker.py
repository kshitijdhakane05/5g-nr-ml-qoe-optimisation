"""Background process entry point for the live ns-3 controller."""
from simulation.live_control import worker_main


if __name__ == "__main__":
    worker_main()
