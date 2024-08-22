import threading
import logging


class TokenBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.current_index = 0
        self.mutex = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_locking(self, token):
        with self.mutex:
            self.buffer[self.current_index] = token
            self.current_index = (self.current_index + 1) % self.size

    def clear_locking(self):
        with self.mutex:
            self.buffer = [None] * self.size

    def __str__(self):
        return "".join(
            filter(
                None,
                self.buffer[self.current_index :] + self.buffer[: self.current_index],
            )
        )

    def get_current_buffer(self):
        return self.buffer
