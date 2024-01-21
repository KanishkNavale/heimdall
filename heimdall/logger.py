import logging

import colorlog


class OverWatch(logging.Logger):
    def __init__(self, name: str = "heimdall") -> None:
        super().__init__(name)

        # Set level to DEBUG to log all levels
        debugger_level = logging.DEBUG
        self.setLevel(debugger_level)

        # Create a file handler
        file_handler = logging.FileHandler("heimdall.log")
        file_handler.setLevel(debugger_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(debugger_level)

        # Create a color formatter
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s %(name)s | %(levelname)s | %(process)s | %(asctime)s | %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
            },
            reset=True,
            style="%",
            datefmt="%d-%m-%Y %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.addHandler(console_handler)

        file_formatter = logging.Formatter(
            "%(levelname)s | %(process)s | %(asctime)s | %(message)s",
            datefmt="%d-%m-%Y | %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)
