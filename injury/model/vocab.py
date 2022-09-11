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

        # import matplotlib.pyplot as plt

        # for indx in range(batch_size):
        #     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        #     ax0.plot(
        #         dt_vals[indx].numpy(),
        #         time_density[indx].detach().numpy(),
        #         linestyle="-",
        #         linewidth=0.8,
        #     )
        #     ax0.set_title(
        #         "Probability density $p_i(u)$\nof the next increment"
        #     )
        #     ax0.set_xlabel("Time $u$")
        #     ax0.set_ylabel("density $p_i(u)$")
        #     ylims = ax0.get_ylim()
        #     ax0.vlines(
        #         estimate_dt[indx].item(),
        #         *ylims,
        #         linestyle="--",
        #         linewidth=0.7,
        #         color="red",
        #         label=r"estimate $\hat{t}_i - t_{i-1}$",
        #     )
        #     ax0.vlines(
        #         dt.data[indx].item(),
        #         *ylims,
        #         linestyle="--",
        #         linewidth=0.7,
        #         color="green",
        #         label=r"true $t_i - t_{i-1}$",
        #     )
        #     ax0.set_ylim(ylims)
        #     ax0.legend()
        #     # ax1.plot(
        #     #     dt_vals[indx].numpy(),
        #     #     pred_lambda[indx].detach().numpy(),
        #     #     linestyle="-",
        #     #     linewidth=0.7,
        #     #     label=r"total intensity $\bar\lambda$",
        #     # )
        #     for k in range(pred_lambda_k.size(2)):
        #         ax1.plot(
        #             dt_vals[indx].numpy(),
        #             pred_lambda_k[indx, :, k].detach().numpy(),
        #             label="type {}".format(k),
        #             linestyle="--",
        #             linewidth=0.7,
        #         )
        #     ax1.set_title(
        #         f"Intensities (previous event = {prev_events[indx]})"
        #     )
        #     ax1.set_xlabel("Time $t$")
        #     ax1.legend()
        #     plt.show()
