"""MeanFlow objective + one-step / few-step samplers.

Interpolant (flow-matching convention):
    z_t = (1 - t) * x0 + t * eps,   eps ~ N(0, I)
    conditional velocity  v = dz_t/dt = eps - x0          (E[v | z_t] = marginal velocity)

Average velocity (what the network u_theta models):
    u(z_t, r, t) = 1/(t - r) * integral_r^t v(z_tau, tau) dtau

MeanFlow identity (differentiate (t-r)u = integral wrt t, holding r fixed):
    u(z_t, r, t) = v(z_t, t) - (t - r) * d/dt u(z_t, r, t)
where the *total* derivative d/dt u = d_z u . v + d_t u  is exactly a JVP of u
along the tangent (v, 0, 1) on inputs (z_t, r, t)  -- r is held fixed so its tangent is 0.

Training: regress u_theta(z,r,t) onto sg[ v - (t-r) * d/dt u ]  (stop-grad on the target).
One-step sampling: x0 = eps - u_theta(eps, r=0, t=1)   (a single network evaluation).
"""
from __future__ import annotations
import torch
import torch.func as func


def sample_r_t(batch, device, dist="uniform", p_eq=0.5, max_gap=1.0, mu=-0.4, sigma=1.0, generator=None):
    """Sample (r, t) with r <= t.

    Two stabilizers for the JVP bootstrap:
      * `p_eq`: fraction of the batch with r == t  -> the (t-r) term vanishes and the
        loss reduces to plain flow matching (learns the instantaneous velocity).
      * `max_gap`: cap on (t - r). The unstable term in the target is (t-r)*du/dt;
        capping (t-r) small early (then ramping it to 1) prevents du/dt from blowing
        up before the velocity field is learned (curriculum)."""
    if dist == "logitnormal":
        t = torch.sigmoid(mu + sigma * torch.randn(batch, device=device, generator=generator))
    else:
        t = torch.rand(batch, device=device, generator=generator)
    gap = torch.rand(batch, device=device, generator=generator) * max_gap
    r = (t - gap).clamp(min=0.0)
    eq = torch.rand(batch, device=device, generator=generator) < p_eq
    r = torch.where(eq, t, r)
    return r, t


def meanflow_loss(model, x0, *, eps=None, dist="uniform", p_eq=0.5, max_gap=1.0, gamma=1.0, c=1e-3):
    """Returns (loss, mean_residual). `gamma`/`c` control the adaptive L2 weight.
    Pass `eps` to use a FIXED noise per data point (fixed coupling).
    `max_gap` caps (t-r) for the curriculum stabilizer."""
    B = x0.shape[0]
    device = x0.device
    if eps is None:
        eps = torch.randn_like(x0)
    r, t = sample_r_t(B, device, dist=dist, p_eq=p_eq, max_gap=max_gap)
    t_b, r_b = t.view(B, 1, 1, 1), r.view(B, 1, 1, 1)

    z = (1.0 - t_b) * x0 + t_b * eps
    v = eps - x0  # conditional velocity sample

    def fn(z_, r_, t_):
        return model(z_, r_, t_)

    # JVP tangent: (v) for z, (0) for r [held fixed], (1) for t.
    tangents = (v, torch.zeros_like(r), torch.ones_like(t))
    u_pred, dudt = func.jvp(fn, (z, r, t), tangents)

    # Target uses the identity; stop-gradient so grads flow only through u_pred.
    u_tgt = (v - (t_b - r_b) * dudt).detach()
    err = u_pred - u_tgt

    # Adaptive ("relative") L2 weighting from the paper: w = 1 / (||err||^2 + c)^gamma,
    # computed per-sample and detached. gamma=1 -> bounded, scale-invariant loss.
    sq = err.pow(2).flatten(1).mean(dim=1)  # (B,)
    with torch.no_grad():
        w = 1.0 / (sq + c).pow(gamma)
    loss = (w * sq).mean()
    return loss, sq.mean().detach()


@torch.no_grad()
def generate(model, n, shape=(3, 32, 32), steps=1, device="cpu", generator=None, clamp=True, eps=None):
    """Few-step sampler. steps=1 is the headline 1-NFE generation.
    Pass `eps` (n,*shape) to sample from specific noise (e.g. the fixed-coupling noises)."""
    z = eps if eps is not None else torch.randn(n, *shape, device=device, generator=generator)
    n = z.shape[0]
    if steps == 1:
        r = torch.zeros(n, device=device)
        t = torch.ones(n, device=device)
        z = z - model(z, r, t)  # (t - r) = 1
    else:
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t = torch.full((n,), ts[i].item(), device=device)
            r = torch.full((n,), ts[i + 1].item(), device=device)
            z = z - (ts[i] - ts[i + 1]).item() * model(z, r, t)
    return z.clamp(-1, 1) if clamp else z
