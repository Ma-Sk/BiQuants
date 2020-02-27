import time
from contextlib import ContextDecorator
from datetime import datetime


class timeit(ContextDecorator):
    def __init__(self, string='Time: {time:.2f} s.'):
        self.string = string

    def __enter__(self):
        print('-' * 30, "\nCurrent Time: ", datetime.now().strftime("%H:%M:%S"))
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.string.format(time=(time.time() - self.start_time)))
