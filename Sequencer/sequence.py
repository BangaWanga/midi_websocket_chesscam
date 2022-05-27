

class Sequence:
    def __init__(self, seq_array):
        self.seq = seq_array
        self.max_steps = len(seq_array)
        self.act_step = 0

    def run(self):
        midi_note = self.seq[self.act_step]
        self.incr_act_step()
        return midi_note

    def reet(self):
        self.act_step = 0

    def incr_act_step(self):
        self.act_step += 1
        self.act_step = self.act_step % self.max_steps


if __name__ == "__main__":
    print("allo")

    seq = Sequence([0,1])
    print(seq.run())
    print(seq.run())
    print(seq.run())