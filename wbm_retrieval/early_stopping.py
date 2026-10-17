"""Validation-only stopping state, independent of a model or compute backend."""

import math


DEFAULTS = {
    "local": {"enabled": False, "patience": 10, "min_delta": 0.001, "min_epochs": 10},
    "global": {"enabled": False, "patience": 5, "min_delta": 0.0001, "min_epochs": 5},
}


class EarlyStopping:
    def __init__(self, enabled=True, patience=5, min_delta=0.0001, min_epochs=1):
        if patience < 1 or min_epochs < 1 or min_delta < 0:
            raise ValueError("patience/min_epochs must be positive and min_delta nonnegative")
        self.enabled = bool(enabled)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.reference = None
        self.bad_epochs = 0
        self.last_epoch = 0
        self.stopped = False

    @classmethod
    def from_settings(cls, settings, kind):
        return cls(**{**DEFAULTS[kind], **settings.get("early_stopping", {})})

    def step(self, value, epoch):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite validation loss at epoch {epoch}: {value}")
        if epoch <= self.last_epoch:
            raise ValueError("Early stopping epochs must be strictly increasing")
        if self.stopped:
            raise ValueError("A stopped training run must not execute another epoch")
        self.last_epoch = epoch
        if self.reference is None or value < self.reference - self.min_delta:
            self.reference = float(value)
            self.bad_epochs = 0
        elif epoch > self.min_epochs:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        self.stopped = bool(self.enabled and epoch > self.min_epochs and self.bad_epochs >= self.patience)
        return self.stopped

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        for key in ("enabled", "patience", "min_delta", "min_epochs"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Early stopping configuration mismatch: {key}")
        for key in ("reference", "bad_epochs", "last_epoch", "stopped"):
            setattr(self, key, state[key])
