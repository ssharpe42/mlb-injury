import torch
from IPython import embed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class PointProcessDataset(Dataset):

    data_types = [
        "times",
        "events",
        "seq_lengths",
        "labels",
        "end_times",
    ]

    def __init__(
        self,
        times,
        events=None,
        event_dim=None,
        labels=None,
        bos=True,
        has_eos=True,
        tmax=0.0,
    ) -> None:
        """Dataset for point processes

        Args:
            times: list of list/array of event times
            events: list of list/array of events (integers). Defaults to None.
            event_dim: number of event types. Defaults to None.
            labels: array of integer labels associated with a sequence.
            bos: create a beginning of sequence event. Defaults to True.
            has_eos: contains end of sequence event. Defaults to False.
            tmax: if eos is false this value doesn't matter since the last event time is used. If tmax is
                  0 then the event time of the last event is used for the eos, otherwise tmax is used. Defaults to 0.
        """
        super().__init__()

        if events is None:
            raise AssertionError(
                "There must be event types for each observation in the sequence."
            )

        if events is not None and event_dim is None:
            raise AssertionError(
                "Must specify the dimension of the event types."
            )
        self.times = list(map(lambda x: torch.tensor(x).float(), times))
        self.seq_lengths = torch.tensor(list(map(len, times)))
        self.events = events
        self.labels = labels
        self.bos = bos
        self.tmax = tmax
        self.has_eos = has_eos

        if events is not None:
            self.event_dim = event_dim
            self.bos_tag = self.event_dim
            # self.eos_tag = self.event_dim - 1 if self.haeos else self.bos_tag
            self.events = list(map(lambda x: torch.tensor(x).long(), events))
            self.end_times = torch.tensor([x[-1] for x in self.times])

        if labels is not None:
            self.labels = torch.tensor(labels)

    def __getitem__(self, index):

        return {
            dtype: getattr(self, dtype)[index]
            for dtype in self.data_types
            if getattr(self, dtype) is not None
        }

    def __len__(self):
        return len(self.events)

    def collate_fn(self, batch):

        batch = {k: [x[k] for x in batch] for k in batch[0]}
        seq_lengths = torch.tensor(batch["seq_lengths"])
        _batch = {}
        # Padding
        _batch["times"] = pad_sequence(
            batch["times"],
            batch_first=True,
            padding_value=0.0 if self.has_eos else self.tmax,
        )
        if "events" in batch:
            _batch["events"] = pad_sequence(
                batch["events"], batch_first=True, padding_value=self.bos_tag
            )

        if "labels" in batch:
            _batch["labels"] = torch.stack(batch["labels"])

        # Append BOS tags
        if self.bos:
            _batch["events"] = torch.cat(
                [
                    self.bos_tag * torch.ones_like(_batch["events"][:, :1]),
                    _batch["events"],
                ],
                dim=1,
            )
            _batch["times"] = torch.cat(
                [torch.zeros_like(_batch["times"][:, :1]), _batch["times"]],
                dim=1,
            )

        # Time between events
        _batch["dt"] = (
            _batch["times"][:, 1:] - _batch["times"][:, :-1]
        ).clamp_min(0)

        if self.has_eos:
            _batch["end_times"] = torch.tensor(batch["end_times"])
            _batch["times"] = torch.concat(
                [
                    _batch["times"][:, 0:1],
                    torch.cumsum(_batch["dt"], dim=1),
                ],
                -1,
            )
        else:
            _batch["end_times"] = self.tmax * torch.ones_like(
                _batch["times"][:, :1]
            )

        # Sort by sequence length
        seq_lengths, indices = seq_lengths.sort(descending=True)

        _batch = {dtype: _batch[dtype][indices] for dtype in _batch}
        _batch["seq_lengths"] = seq_lengths

        return _batch
