"""Entry point."""
from window_tello.app import WindowPassApp
from window_tello.config import AppConfig


def main():
    cfg = AppConfig()
    app = WindowPassApp(cfg)
    app.run()


if __name__ == "__main__":
    main()