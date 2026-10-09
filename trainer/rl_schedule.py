"""Learning-rate schedule and plateau-triggered decay for the Qwen GRPO runs.

A cosine schedule fixes the run length before training starts. Warmup-stable-decay (WSD) holds the peak learning rate
and only decays over the last `decay_steps`, so the decay can be started when the run stops improving.
`PlateauController` makes that call from the periodic evaluations.
"""
import math
import statistics


def scheduled_lr(step, peak, warmup_steps, schedule_start, decay_start, decay_steps):
    """Linear (re)warmup from `schedule_start`, then cosine from `decay_start` down to 10% over `decay_steps`.
    Cosine over a whole run is decay_start=0, decay_steps=max_steps."""
    warmup = min(1., max(step - schedule_start, 0) / max(warmup_steps, 1))
    progress = min(max(step - decay_start, 0) / max(decay_steps, 1), 1)
    return peak * warmup * (.1 + .9 * (1 + math.cos(math.pi * progress)) / 2)


class PlateauController:
    """Chooses the step where a WSD run starts its final decay.

    At each evaluation it gets the validation reward, plus the mean training reward and mean KL over the steps since the
    previous evaluation. A small judged validation set is noisy, so its reward is averaged over the last `window`
    evaluations. Training prompts are new every step, so the windowed training reward is a second, larger held-out
    signal. The run has plateaued when neither signal beats its best by more than `min_delta` for `patience`
    evaluations in a row, counted only from `min_steps`. KL keeps growing while the learning rate is non-zero, so it is
    not a convergence signal; it is a safety ceiling that also starts the decay.

    Calibrated by replaying the 400-step Qwen3-1.7B run: the defaults start the decay at step 350. patience=2 fires at
    step 250, on validation noise.
    """

    STATE = ("val_history", "best_val", "best_train", "stale", "decay_start", "reason")

    def __init__(self, patience=3, min_delta=0.05, window=3, min_steps=100, kl_ceiling=2e-2):
        self.patience, self.min_delta, self.window = patience, min_delta, window
        self.min_steps, self.kl_ceiling = min_steps, kl_ceiling
        self.val_history = []
        self.best_val = self.best_train = -math.inf
        self.stale = 0
        self.decay_start = None
        self.reason = None

    def update(self, step, val_reward, train_reward, kl):
        self.val_history.append(val_reward)
        smoothed = statistics.mean(self.val_history[-self.window:])
        improved = False
        if smoothed > self.best_val + self.min_delta:
            self.best_val, improved = smoothed, True
        if train_reward > self.best_train + self.min_delta:
            self.best_train, improved = train_reward, True
        self.stale = 0 if improved else self.stale + 1
        if self.decay_start is None and step >= self.min_steps:
            if self.kl_ceiling and kl > self.kl_ceiling:
                self.decay_start, self.reason = step, f"KL {kl:.3g} above ceiling {self.kl_ceiling:g}"
            elif self.stale >= self.patience:
                self.decay_start, self.reason = step, f"no gain above {self.min_delta:g} for {self.stale} evaluations"
        return {"plateau/val_smoothed": smoothed, "plateau/train_window": train_reward, "plateau/kl_window": kl,
                "plateau/stale_evals": self.stale, "plateau/decaying": float(self.decay_start is not None)}

    def state_dict(self):
        return {name: getattr(self, name) for name in self.STATE}

    def load_state_dict(self, state):
        for name in self.STATE:
            setattr(self, name, state[name])
