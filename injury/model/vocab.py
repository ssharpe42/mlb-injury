import pickle

import numpy as np


class Vocab:
    def __init__(self, tokens, special_tokens=[]):

        self.token_to_id = {}
        if len(special_tokens) > 0:
            self.token_to_id.update(
                {t: i for i, t in enumerate(special_tokens)}
            )
            tokens = [t for t in tokens if t not in special_tokens]

        self.token_to_id.update(
            {t: i + len(special_tokens) for i, t in enumerate(tokens)}
        )
        self._set_reverse()

    def _set_reverse(self):
        self.id_to_token = {i: b for b, i in self.token_to_id.items()}

    def __len__(self):
        return len(self.token_to_id)

    @property
    def vocab(self):
        return list(self.token_to_id.keys())

    def get_id(self, token):
        return self.token_to_id[token]

    def get_token(self, id):
        return self.id_to_token[id]

    def get_ids(self, tokens):
        return np.array([self.get_id(token) for token in tokens])

    def get_tokens(self, ids):
        return np.array([self.get_token(id) for id in ids])

    def __call__(self, x, reverse=False):
        if reverse:
            return self.get_tokens(x)
        return self.get_ids(x)

    def save(self, path):
        pickle.dump(self.token_to_id, open(path, "wb"))

    def load(self, path):
        self.token_to_id = pickle.load(open(path, "rb"))
        self._set_reverse()
