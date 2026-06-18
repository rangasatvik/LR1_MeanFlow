# MeanFlow — one-step pixel generation (PyTorch)

A from-scratch PyTorch port of the **MeanFlow** objective for one-step image generation
(Geng et al., 2025, *Mean Flows for One-step Generative Modeling*, `arXiv:2505.13447`),
ported from the official JAX/TPU reference. Verified by overfitting 3 images and reproducing
them with a single network evaluation.

**Results**
- **Required — overfit-3 (fixed coupling), 32×32:** one-step (1-NFE) generation reproduces the
  3 images at mean MSE **2.6e-3** (visually identical). → `runs/fix32/`
- **Bonus — random coupling (true generation), 16×16:** one-step samples from random noise show
  recognizable training images. → `runs/rand16/`
- **Correctness:** `python test_meanflow.py` passes 5 checks (incl. JVP vs finite differences).

## 📄 Report

**[▶ View the rendered report](https://htmlpreview.github.io/?https://github.com/rangasatvik/LR1_MeanFlow/blob/main/report.html)** — the full write-up: MeanFlow explained in plain terms, the debugging story (the heavily-weighted part), the result images, and loss curves.

> GitHub displays `.html` files as source code, so the link above renders `report.html` via *htmlpreview*. Alternatives: download [`report.html`](report.html) and open it in any browser, or enable **GitHub Pages** (repo Settings → Pages → Deploy from branch → `main` / root) to host it at `https://rangasatvik.github.io/LR1_MeanFlow/report.html`.

> The build machine had **no GPU** (4-core ARM CPU), so runs use a small model and a resumable,
> time-budgeted trainer. The required overfit is at 32×32; the random-coupling *generation* demo is
> at 16×16 (a tiny CPU-sized model shows clearer modes there). The same core runs at 32/64 px on a
> GPU via `train_imagenette.py`.

## What MeanFlow is (1 paragraph)
Flow matching learns the instantaneous velocity `v(z_t,t)` and integrates an ODE over many steps.
MeanFlow learns the **average velocity** `u(z_t,r,t) = 1/(t−r) ∫_r^t v dτ`, so one step generates:
`x₀ = x₁ − u(x₁,0,1)`. It is trained with the **MeanFlow identity** `u = v − (t−r)·d/dt u`, where
`d/dt u = ∂_t u + (∂_z u)·v` is a JVP with tangent `(v, 0, 1)` on `(z_t, r, t)`, and the target is
stop-gradient'd.

## Files
| file | purpose |
|---|---|
| `model.py` | tiny pixel-space U-Net; Fourier time embedding for `r` and `t`; GroupNorm (JVP-safe) |
| `meanflow.py` | MeanFlow loss (JVP target, stop-grad, adaptive weight), `(r,t)` sampler, 1-/2-step samplers |
| `train_overfit.py` | overfit-3 trainer (fixed/random coupling), resumable + wall-clock budget |
| `train_imagenette.py` | **bonus**: 10-class (CIFAR-10 / Imagenette) GPU training |
| `data.py` | 3 fixed images (scikit-image; procedural fallback) |
| `utils.py` | image grids, loss-curve plot, EMA |
| `test_meanflow.py` | runnable correctness checks (JVP vs finite-diff, grad flow, training, samplers) |
| `build_report.py` | assembles the self-contained `report.html` |
| `REVIEW_PROMPT.md` | brief for an automated code review |

## Install
```bash
pip install -r requirements.txt
```

## Reproduce
```bash
python test_meanflow.py            # correctness checks (~20s, CPU)

# Required: overfit-3, fixed coupling @ 32x32 (the headline result)
python train_overfit.py --coupling fixed --size 32 --tile 2 \
    --p_eq 0.8 --gamma 1.0 --gap0 0.3 --gap_ramp 0.7 \
    --lr 1.5e-3 --clip 0.5 --max_steps 700 --budget 100000

# Bonus: random coupling (true one-step generation)
python train_overfit.py --coupling random --size 16 --tile 4 \
    --p_eq 0.8 --gamma 1.0 --max_steps 800 --budget 100000

python build_report.py             # -> report.html (self-contained)
```
Outputs (targets, 1-/2-step samples, `loss_curve.png`, `metrics.json`) land in `runs/<name>/`.
`--budget` caps per-invocation wall-clock so training fits a 45 s command limit and auto-resumes
from `runs/<name>/ckpt.pt`; set it huge to train in one go.

### Bonus (GPU)
```bash
python train_imagenette.py --data cifar10 --size 32 --base 128 --steps 60000 --bs 128
# or Imagenette:  --data /path/to/imagenette2/train --size 64
```

## The recipe that made it stable (see report for the why)
The JVP bootstrap target `(t−r)·du/dt` blows up at the `(r=0,t=1)` corner. What fixed it:

| knob | value | reason |
|---|---|---|
| `p_eq` (fraction `r=t`) | 0.8 | anchors the instantaneous velocity so `du/dt` stays near true |
| gap curriculum | `(t−r)` cap 0.3 → 1.0 | bounds the unstable term until the field is learned |
| adaptive weight `γ` | 1.0 | down-weights high-residual samples |
| grad clip / lr | 0.5 / 1e-3 | prevents the feedback blow-up |
| norm | **GroupNorm** | BatchNorm breaks `torch.func.jvp` |
| time embedding | Fourier on [0,1] | the classic diffusion embedding aliases continuous `t` |

## Notes
- Verified on PyTorch 2.4.1 (CPU). `torch.func.jvp` returns a primal that still backprops to params,
  so one JVP call yields both the prediction and the (detached) target.
- `runs/<name>/metrics.json` records `reproduction_mse_per_image` (~2.6e-3 for fixed @ 32×32).
