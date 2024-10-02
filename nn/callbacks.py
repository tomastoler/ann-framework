import time
from nn.types import Callback


class Timer(Callback):
    
    def __init__(self):
        pass
    
    def on_epochs_begin(self):
        self.start_time = time.time()
        
    def on_epochs_end(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"Time taken: {duration} seconds")