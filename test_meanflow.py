"""Runnable correctness checks for the MeanFlow implementation.

    python test_meanflow.py      # CPU, ~20s; exits non-zero on any failure

Checks: the JVP total-derivative (the heart of the objective), the (r,t) sampler
invariants, gradient flow through the JVP primal, that training reduces the residual,
and that the samplers produce valid output.
"""
import torch
import torch.func as func
from model import UNet
from meanflow import meanflow_loss, generate, sample_r_t


def main():
    torch.manual_seed(0)

    # ---- 1) JVP total-derivative d/dt u matches finite differences along (v, 0, 1) ----
    net = UNet(base=24, ch_mult=(1, 2), num_res=1, emb_dim=128)
    with torch.no_grad():  # output conv is zero-initialized by design; make it nonzero
        net.out_conv.weight.normal_(0, 0.1)
        net.out_conv.bias.normal_(0, 0.1)
    net.eval()
    z = torch.randn(2, 3, 16, 16); r = torch.rand(2); t = torch.rand(2); v = torch.randn_like(z)

    def fn(z_, r_, t_):
        return net(z_, r_, t_)

    _, dudt = func.jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    h = 1e-3
    fd = (fn(z + h * v, r, t + h * torch.ones_like(t)) - fn(z, r, t)) / h
    rel = ((fd - dudt).norm() / dudt.norm()).item()
    assert rel < 5e-2, f"JVP != finite difference (rel err {rel:.3f})"
    print(f"[PASS] JVP total-derivative matches finite differences (rel err {rel:.2e})")

    # ---- 2) (r,t) sampler: r<=t, gap cap honored, p_eq fraction correct ----
    r2, t2 = sample_r_t(8192, "cpu", p_eq=0.3, max_gap=0.4)
    assert (r2 <= t2 + 1e-6).all(), "r > t produced"
    assert ((t2 - r2) <= 0.4 + 1e-6).all(), "max_gap violated"
    frac_eq = (r2 == t2).float().mean().item()
    assert 0.2 < frac_eq < 0.4, f"p_eq fraction off ({frac_eq:.2f})"
    print(f"[PASS] sample_r_t: r<=t, gap<=0.4, r==t fraction {frac_eq:.2f}")

    # ---- 3) gradient flows through the JVP primal to every parameter ----
    net.train()
    loss, _ = meanflow_loss(net, torch.randn(6, 3, 16, 16), gamma=0.0)
    loss.backward()
    ng = sum(p.grad is not None for p in net.parameters())
    nt = sum(1 for _ in net.parameters())
    assert ng == nt, f"only {ng}/{nt} params received gradient"
    print(f"[PASS] gradient flows through JVP primal to all {nt} param tensors")

    # ---- 4) training reduces the residual (tiny fixed-coupling overfit) ----
    net2 = UNet(base=24, ch_mult=(1, 2), num_res=1, emb_dim=128)
    opt = torch.optim.Adam(net2.parameters(), lr=2e-3)
    x = torch.randn(3, 3, 16, 16); eps = torch.randn_like(x)
    first = None
    for i in range(60):
        opt.zero_grad(set_to_none=True)
        l, res = meanflow_loss(net2, x.repeat(2, 1, 1, 1), eps=eps.repeat(2, 1, 1, 1),
                               p_eq=0.8, max_gap=0.3, gamma=0.0)
        l.backward(); opt.step()
        if i == 0:
            first = float(res)
    assert float(res) < first, f"residual did not decrease ({first:.3f} -> {float(res):.3f})"
    print(f"[PASS] training reduces residual ({first:.3f} -> {float(res):.3f})")

    # ---- 5) samplers produce valid shapes / range ----
    s1 = generate(net2, 4, shape=(3, 16, 16), steps=1)
    s2 = generate(net2, 4, shape=(3, 16, 16), steps=2)
    assert s1.shape == (4, 3, 16, 16) and s2.shape == (4, 3, 16, 16)
    assert s1.min() >= -1.0001 and s1.max() <= 1.0001
    print("[PASS] samplers: correct shapes and value range")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
