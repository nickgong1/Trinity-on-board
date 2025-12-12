import torch
from torchdiffeq import odeint


class Sampler:
    """Self-contained ODE-only sampler for velocity-predicting models.
    
    Compatible construction:
      - Sampler(transport)  # extracts sample_eps
      - Sampler(sample_eps=0.0)
    """
    def __init__(
        self,
    ):
        pass
        

    def _interval(self, *, reverse=False):
        # Velocity-only: stable across the whole interval; avoid the endpoint at t=1 by epsilon
        t0 = 0.0
        t1 = 1.0
        if reverse:
            t0, t1 = 1.0 - t0, 1.0 - t1
        return t0, t1

    def sample_ode(
        self,
        *,
        sampling_method="euler",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """Return a sampling function that integrates the probability flow ODE."""
        t0, t1 = self._interval(reverse=reverse)
        t_vec = torch.linspace(t0, t1, num_steps)

        def sample(x, model, **model_kwargs):
            device = x[0].device if isinstance(x, tuple) else x.device
            t = t_vec.to(device)

            def _fn(t_scalar, state):
                batch = state[0].size(0) if isinstance(state, tuple) else state.size(0)
                t_batch = torch.ones(batch, device=device) * t_scalar
                return model(state, t_batch, **model_kwargs)

            ### Only matters for adaptive sampling method (e.g. dopri5)
            atol_list = [atol] * (len(x) if isinstance(x, tuple) else 1)
            rtol_list = [rtol] * (len(x) if isinstance(x, tuple) else 1)

            samples = odeint(
                _fn,
                x,
                t,
                method=sampling_method,
                atol=atol_list,
                rtol=rtol_list,
            )
            return samples

        return sample


