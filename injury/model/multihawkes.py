import os

import pytorch_lightning as pl
import torch
from einops import rearrange, reduce, repeat
from IPython import embed
from torch import nn
from torchmetrics import AUROC, AveragePrecision, F1Score

from .base_models import PointProcessModel


class ExponentialKernel(nn.Module):
    def __init__(
        self,
        event_dim,
    ):
        super(ExponentialKernel, self).__init__()
        self.event_dim = event_dim
        self.log_alpha = nn.Embedding(self.event_dim + 1, self.event_dim + 1)
        self.log_delta = nn.Embedding(self.event_dim + 1, self.event_dim + 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, -5.0)

    def forward(self, events):
        """Forward pass of ExponentialKernel

        Args:
            events: event types
        """

        alphas = self.log_alpha(events).exp()
        deltas = self.log_delta(events).exp()
        return alphas, deltas

    def lambda_k(self, time_to_current, mask, decay_params):

        alphas, deltas = decay_params

        kernel_lambda_k = (
            rearrange(alphas, "b n d -> b 1 n d")
            * torch.exp(
                -rearrange(deltas, "b n d -> b 1 n d")
                * (time_to_current * mask)
            )
            * mask
        ).sum(2)

        return kernel_lambda_k

    def integral_k(self, time_to_end, decay_params):

        alphas, deltas = decay_params
        kernel_integral_k = (
            (alphas / deltas)
            * (1 - torch.exp(-deltas * time_to_end[:, :, None]))
        ).sum(1)
        return kernel_integral_k


class ConstantKernel(nn.Module):
    def __init__(
        self,
        event_dim,
    ):
        super(ConstantKernel, self).__init__()
        self.event_dim = event_dim

    def forward(self, events):
        """Forward pass of ConstantKernel

        Args:
            events: event types
        """
        return torch.zeros((*events.shape, self.event_dim + 1))

    def lambda_k(self, time_to_current, mask, decay_params):

        zero_ = decay_params
        kernel_lambda_k = (0 * time_to_current).sum(2)
        return kernel_lambda_k

    def integral_k(self, time_to_end, decay_params):

        zero_ = decay_params
        kernel_integral_k = (0 * time_to_end[:, :, None]).sum(1)
        return kernel_integral_k


class RayleighKernel(nn.Module):
    def __init__(
        self,
        event_dim,
    ):
        super(RayleighKernel, self).__init__()
        self.event_dim = event_dim
        self.log_sigma = nn.Embedding(self.event_dim + 1, self.event_dim + 1)

    def forward(self, events):
        """Forward pass of ExponentialKernel

        Args:
            events: event types
        """
        sigmas = self.log_sigma(events).exp()
        return sigmas

    def lambda_k(self, time_to_current, mask, decay_params):

        sigmas = decay_params
        inv_sigma = 1 / rearrange(sigmas, "b n d -> b 1 n d")
        kernel_lambda_k = (
            (
                time_to_current
                * inv_sigma
                * torch.exp(-(time_to_current ** 2) * 0.5 * inv_sigma * mask)
            )
            * mask
        ).sum(2)

        return kernel_lambda_k

    def integral_k(self, time_to_end, decay_params):

        sigmas = decay_params
        inv_sigma = 1 / sigmas
        kernel_integral_k = (
            1 - torch.exp(-time_to_end[:, :, None] ** 2 * 0.5 * inv_sigma)
        ).sum(1)
        return kernel_integral_k


class MultivariateHawkes(PointProcessModel):
    def __init__(
        self,
        kernel="exponential",
        vocab=None,
        *args,
        **kwargs,
    ):
        super(MultivariateHawkes, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.vocab = vocab
        self.log_mu = nn.Embedding(self.event_dim + 1, 1)

        if kernel == "exponential":
            self.decay = ExponentialKernel(self.event_dim)
        elif kernel == "constant":
            self.decay = ConstantKernel(self.event_dim)
        elif kernel == "rayleigh":
            self.decay = RayleighKernel(self.event_dim)

        nn.init.uniform_(self.log_mu.weight, -5.0, -5.0)

    def reset_parameters(self):
        nn.init.uniform_(self.log_mu.weight, -5.0, -5.0)

    def forward(self, events):
        """Forward pass of MultivariateHawkes

        Args:
            events: event types
        """

        mus = self.log_mu(events).exp()
        decay_params = self.decay(events)

        return mus, decay_params

    def _log_event_intensities(self, events, seq_lengths, lambda_k):
        # (batch_size, seq_lengths, event_dim )
        event_onehot = torch.eye(self.event_dim + 1)[events]
        event_indx = rearrange(torch.arange(events.shape[1]), "n -> 1 n 1")
        seq_lengths_ = rearrange(seq_lengths, "b -> b 1 1")

        # Keep all events that are not BOS/EOS
        seq_mask = (event_indx < seq_lengths_) & (event_indx > 0)

        lambda_k_log = lambda_k.log() * event_onehot * seq_mask

        sum_log_intensities = lambda_k_log.sum((2, 1))
        return sum_log_intensities

    def _lambda_k(self, inputs, times, events):
        mus, decay_params = inputs
        time_to_current = rearrange(times, "b n -> b n 1 1") - rearrange(
            times, "b n -> b 1 n 1"
        )
        mask = time_to_current > 0
        kernel_lambda_k = self.decay.lambda_k(
            time_to_current, mask, decay_params
        )

        lambda_k = mus + kernel_lambda_k

        return lambda_k

    def _integral_k(self, inputs, times, end_times):

        _, decay_params = inputs
        time_to_end = end_times - times
        kernel_integral_k = self.decay.integral_k(time_to_end, decay_params)
        mu_integral_k = self.log_mu.weight.exp().transpose(1, 0) * end_times

        integral_k = mu_integral_k + kernel_integral_k

        return integral_k

    def _integral(self, integral_k):
        return integral_k.sum(1)

    def neg_log_likelihood(
        self, events, dt, times, end_times, seq_lengths, output
    ):
        """Negative log likelihood calculation for Hawkes process

        Args:
            events: event types
            times: time of each event
            end_times: end times of each sequence
            seq_lengths: lengths of sequences
            output: tuple of all outputs from forward (mus, alphas, deltas)


        Returns:
            sum of negative log likelihood for batch
        """
        # (batch_size, max seq_lenth, event_dim)
        lambda_k = self._lambda_k(inputs=output, times=times, events=events)
        # (batch size,)
        sum_log_intensities = self._log_event_intensities(
            events, seq_lengths, lambda_k
        )
        # (batch_size, event_dim)
        integral_k = self._integral_k(
            output, times=times, end_times=end_times.unsqueeze(1)
        )
        # (batch_size, )
        integral = self._integral(integral_k)
        nll = -sum_log_intensities + integral
        return nll.sum()

    def predict_all(
        self,
        prev_events,
        prev_times,
        prev_dt,
        seq_lengths,
        max_dt=10000,
        mc_samples=2000,
        return_lambdas=False,
    ):
        # pass
        with torch.no_grad():
            _, decay_params = self.forward(prev_events)

        # sample_dt = torch.linspace(
        #     max_dt / mc_samples, max_dt, mc_samples
        # ).repeat(len(prev_events), 1)

        sample_dt = torch.linspace(
            0, max_dt - max_dt / mc_samples, mc_samples
        ).repeat(len(prev_events), 1)

        # prediction mask excluding BOS/EOS
        pred_mask = rearrange(
            torch.arange(1, prev_events.shape[1]), "n -> 1 n"
        ) < rearrange(seq_lengths, "b -> b 1 ")

        sample_times_all = prev_times.unsqueeze(-1) + sample_dt.unsqueeze(1)

        dt_preds = []
        event_preds = []
        event_probs = []
        n_pred = prev_events.shape[1] - 1

        if return_lambdas:
            ret_lambdas = []
            n_pred = prev_events.shape[1]

        for i in range(n_pred):

            sample_times = sample_times_all[:, i]
            time_to_current = rearrange(
                sample_times, "b s -> b s 1 1"
            ) - rearrange(prev_times[:, : i + 1], "b n -> b 1 n 1")

            seq_mask = rearrange(
                torch.arange(i + 1), "n -> 1 1 n 1"
            ) < rearrange(seq_lengths, "b -> b 1 1 1")
            decay_params_ = [x[:, : (i + 1)] for x in decay_params]
            kernel_lambda_k = self.decay.lambda_k(
                time_to_current, seq_mask, decay_params_
            )

            mus = self.log_mu.weight.exp().transpose(1, 0)
            pred_lambda_k = mus + kernel_lambda_k
            pred_lambda = pred_lambda_k.sum(2)

            if return_lambdas:
                if i < n_pred - 1:
                    before_next_event = sample_times < prev_times[:, i + 1]
                    pred_lambda_k = pred_lambda_k[before_next_event, :]
                # if i == n_pred - 1:
                #     # embed()
                ret_lambdas.append(pred_lambda_k.squeeze())

            else:
                timestep = max_dt / mc_samples
                integral_ = (timestep * pred_lambda).cumsum(1)
                time_density = pred_lambda * torch.exp(-integral_)

                lambda_ratio = pred_lambda_k / pred_lambda.unsqueeze(-1)
                time_integrand = sample_dt * time_density  # t*p_i(t)
                event_integrand = lambda_ratio * time_density.unsqueeze(-1)

                # Trapezoid method
                estimate_dt = (
                    timestep
                    * 0.5
                    * (time_integrand[:, 1:] + time_integrand[:, :-1])
                ).sum(1)
                event_prob = (
                    timestep
                    * 0.5
                    * (event_integrand[:, 1:] + event_integrand[:, :-1])
                ).sum(1)
                event_pred = torch.argmax(event_prob, dim=1)

                event_preds.append(event_pred)
                event_probs.append(event_prob)
                dt_preds.append(estimate_dt)

        if return_lambdas:
            return ret_lambdas

        event_preds = torch.vstack(event_preds).transpose(0, 1)
        event_probs = torch.stack(event_probs).transpose(0, 1)
        dt_preds = torch.vstack(dt_preds).transpose(0, 1)

        return (
            event_probs[pred_mask],
            event_preds[pred_mask],
            prev_events[:, 1:][pred_mask],
            dt_preds[pred_mask],
            prev_dt[pred_mask],
        )

    def training_step(self, batch, batch_idx):
        loss, event_num = self._step(batch, batch_idx)
        self.log("train_loss", loss / event_num)
        return {"loss": loss, "event_num": event_num}

    def validation_step(self, batch, batch_idx):
        loss, event_num = self._step(batch, batch_idx, evaluate=True)
        return {"val_loss": loss, "event_num": event_num}

    def validation_epoch_end(self, outputs):
        total_loss = torch.stack([step["val_loss"] for step in outputs]).sum()
        event_num = torch.stack([step["event_num"] for step in outputs]).sum()
        loss = total_loss / event_num
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        (
            loss,
            event_num,
            event_probs,
            event_pred,
            events,
            estimate_dt,
            dt,
        ) = self._step(
            batch,
            batch_idx,
            evaluate=True,
            predict_next=True,
        )
        return {
            "test_loss": loss,
            "event_num": event_num,
            "event_probs": event_probs,
            "event_pred": event_pred,
            "events": events,
            "estimate_dt": estimate_dt,
            "dt": dt,
        }

    def test_epoch_end(self, outputs):
        total_loss = torch.stack([step["test_loss"] for step in outputs]).sum()
        event_num = torch.stack([step["event_num"] for step in outputs]).sum()
        event_probs = torch.vstack([step["event_probs"] for step in outputs])
        event_pred = torch.cat([step["event_pred"] for step in outputs])
        events_true = torch.cat([step["events"] for step in outputs])
        estimate_dt = torch.cat([step["estimate_dt"] for step in outputs])
        dt_true = torch.cat([step["dt"] for step in outputs])

        num_classes = event_probs.shape[1]

        event_auroc = AUROC(num_classes=num_classes, average="weighted")(
            event_probs, events_true
        ).item()
        event_aupr = AveragePrecision(num_classes, average="weighted")(
            event_probs, events_true
        ).item()
        event_acc = (event_pred == events_true).float().mean()
        # embed()
        # (events_true.unsqueeze(-1) == torch.topk(event_probs, 3).indices).any(
        #     1
        # ).float().mean()
        dt_rmse = ((estimate_dt - dt_true) ** 2).mean().sqrt().item()
        loss = (total_loss / event_num).item()

        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_dt_rmse", dt_rmse, on_epoch=True, on_step=False)
        self.log("test_acc", event_acc, on_epoch=True, on_step=False)
        self.log("test_auroc", event_auroc, on_epoch=True, on_step=False)
        self.log("test_aupr", event_aupr, on_epoch=True, on_step=False)
        return {
            "test_loss": loss,
            "event_aupr": event_aupr,
            "dt_rmse": dt_rmse,
            "event_acc": event_acc,
            "event_auroc": event_auroc,
        }

    def _step(self, batch, batch_idx, evaluate=False, predict_next=False):

        output = self.forward(batch["events"])
        loss = self.neg_log_likelihood(
            batch["events"],
            batch["dt"],
            batch["times"],
            batch["end_times"],
            batch["seq_lengths"],
            output,
        )
        seq_lengths = batch["seq_lengths"].sum().float()
        if not predict_next:
            return loss, seq_lengths

        event_probs, event_pred, events, dt_pred, dt = self.predict_all(
            batch["events"],
            batch["times"],
            batch["dt"],
            batch["seq_lengths"],
        )

        return loss, seq_lengths, event_probs, event_pred, events, dt_pred, dt
