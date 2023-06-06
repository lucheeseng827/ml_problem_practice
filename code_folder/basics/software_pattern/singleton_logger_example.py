import logging

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Additional logger configuration goes here

    def log(self, message):
        self.logger.info(message)



logger = Logger()

logger.log("Logging a message")  # Log a message using the singleton logger instance
# Output: INFO:__main__:Logging a message
