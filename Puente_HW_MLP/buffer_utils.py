import numpy as np


class RingBuffer2Ch:
    def __init__(self, size: int):
        self.size = size
        self.ch1 = np.zeros(size, dtype=np.float32)
        self.ch2 = np.zeros(size, dtype=np.float32)
        self.index = 0
        self.is_full = False

    def append(self, x1: float, x2: float) -> None:
        self.ch1[self.index] = x1
        self.ch2[self.index] = x2
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.is_full = True

    def get_ordered(self):
        if not self.is_full:
            return self.ch1[:self.index].copy(), self.ch2[:self.index].copy()

        ch1_ordered = np.concatenate((self.ch1[self.index:], self.ch1[:self.index]))
        ch2_ordered = np.concatenate((self.ch2[self.index:], self.ch2[:self.index]))
        return ch1_ordered, ch2_ordered

    def __len__(self):
        return self.size if self.is_full else self.index