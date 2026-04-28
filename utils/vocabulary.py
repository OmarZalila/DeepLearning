from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold

        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        self.itos = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token,
        }
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text: str):
        return [self.stoi.get(token, self.stoi[self.unk_token]) for token in text.split()]
