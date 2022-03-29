import numpy as np


class Track:
    # TODO: Try variations like patterns with different lengths (maybe on smaller raster)
    def __init__(self, n_steps: int = 16, n_sequences: int = 12):
        self.n_steps = n_steps
        self.n_sequences = n_sequences
        self.sequences = np.zeros((self.n_steps, self.n_sequences), dtype=np.int32)

    def update(self, sequences: np.ndarray) -> bool:
        if np.array_equal(self.sequences, sequences):
            return False
        else:
            print("Sequences have changed")
            self.sequences = sequences
            return True

    def update_deprecated(self, sequences: np.ndarray) -> bool:
        new_sequences = np.zeros((self.n_steps, self.n_sequences), dtype=np.int32)
        for i, seq in enumerate(sequences):
            for j, val in enumerate(seq):
                if val:  # if the val is not zero
                    # TODO: why?
                    # The encoding works like this:
                    # The sequences 0 to 2 belong to the first two rows on the chessboard.
                    # The sequences 3 to 5 belong to the third and fourth row on the chessboard.
                    # ...
                    # The vals range from 0 to 3: 0 is off, 1 to 3 (red, green, blue) are the sub-voices
                    new_sequences[3*i + val - 1, j] = 1
        if np.array_equal(self.sequences, new_sequences):
            return False
        else:
            print("New Ones")
            print(new_sequences)
            self.sequences = new_sequences
            return True


