import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        prediction_type: str = "velocity",
    ) -> torch.Tensor:
        """Compute training target.

        Args:
            prediction_type: "velocity" → target = noise - sample (flow matching velocity)
                             "sample"   → target = sample (x₀ prediction)
        """
        del timestep
        if prediction_type == "velocity":
            return noise - sample
        elif prediction_type == "sample":
            return sample
        else:
            raise ValueError(f"Unknown prediction_type '{prediction_type}', expected 'velocity' or 'sample'")

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    def step(
        self,
        model_output: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        prediction_type: str = "velocity",
        is_last_step: bool = False,
    ) -> torch.Tensor:
        """Single Euler step.

        Args:
            model_output: model prediction (velocity or x₀ depending on prediction_type)
            delta:        Δσ for this step (negative, moves from σ toward 0)
            sample:       current noisy sample z_t
            timestep:     current timestep (required for prediction_type="sample")
            prediction_type: "velocity" → model_output is velocity v, step directly
                             "sample"   → model_output is x̂₀, convert to velocity first:
                                          v = (z_t - x̂₀) / σ, then z_{t+1} = z_t + v * Δσ
            is_last_step:   when True and prediction_type="sample", return x̂₀ directly
                            instead of Euler step (avoids 1/σ amplification at σ→0)
        """
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim != 0:
            delta = delta.view(-1, *([1] * (sample.ndim - 1)))

        if prediction_type == "velocity":
            return sample + model_output * delta
        elif prediction_type == "sample":
            # Last step: return x̂₀ directly (no 1/σ amplification)
            if is_last_step:
                return model_output
            if timestep is None:
                raise ValueError("`timestep` is required for prediction_type='sample'")
            sigma = (timestep / float(self.num_train_timesteps)).to(
                sample.device, dtype=sample.dtype
            )
            if sigma.ndim != 0:
                sigma = sigma.view(-1, *([1] * (sample.ndim - 1)))
            sigma = sigma.clamp(min=1e-5)
            velocity = (sample - model_output) / sigma
            return sample + velocity * delta
        else:
            raise ValueError(f"Unknown prediction_type '{prediction_type}'")
