import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from injury.model.multihawkes import MultivariateHawkes
from IPython import embed
from sklearn.model_selection import train_test_split

from .vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--position", default=None, required=False, type=str)
parser.add_argument("-v", "--version", default=None, required=False, type=str)
parser.add_argument(
    "-d", "--dtd", action="store_true", help="split dtd injuries"
)
parser.add_argument("-t", "--test_path", type=str)
args = parser.parse_args()


data = pd.read_parquet("injury_final.parquet").query(
    "injury_location!='[START]'"
)

if args.position is not None:
    data = data[data["position"] == args.position]

if args.dtd:
    data["injury_location"] = np.where(
        data["injury_location"] == "[END]",
        "[END]",
        data["injury_location"]
        + data["dtd"].map({True: " (dtd)", False: "", np.nan: "", None: ""}),
    )

vocab = Vocab(np.sort(data.injury_location.unique()), special_tokens=["[END]"])
data["injury_ids"] = vocab(data["injury_location"])
seq = (
    data.groupby(["player_id", "name"], as_index=False)[["injury_ids", "t"]]
    .agg(list)
    .rename(columns={"injury_ids": "events", "t": "times"})
)

train = seq[["events", "times"]]
test = train.sample(50)
# train, test = train_test_split(
#     seq[["events", "times"]], test_size=0.15, random_state=42
# )

pl.seed_everything(42)

if args.test_path is not None:
    nh = MultivariateHawkes.load_from_checkpoint(args.test_path).eval()

else:
    nh = MultivariateHawkes(
        event_dim=len(vocab),
        has_eos=True,
        bos=True,
        kernel="exponential",
    )

nh.prepare_data(
    train=train.to_dict(orient="list"),
    dev=train.to_dict(orient="list"),
    test=test.to_dict(orient="list"),
)

tb_logger = pl.loggers.TensorBoardLogger(
    save_dir="logs/", version=args.version
)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    verbose=False,
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", mode="min", patience=10, min_delta=0.005
)
trainer = pl.Trainer(
    max_epochs=1000,
    logger=tb_logger,
    callbacks=[checkpoint_callback, early_stopping],
)
if args.test_path is None:
    trainer.fit(nh)
    vocab.save(os.path.join(trainer.log_dir, "vocab.pkl"))

trainer.test(nh)
