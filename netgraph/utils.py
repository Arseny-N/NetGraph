from itertools import tee

class IterWithLen:
    def __init__(self, iter, len):
        self.len = len
        self.iter = iter
    def __len__(self):
        return self.len
    def __iter__(self):
        self.iter, iter = tee(self.iter)
        return iter