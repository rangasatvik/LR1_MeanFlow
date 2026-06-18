# Review prompt for Claude Code

Copy everything below the line into Claude Code (run it from inside this repo folder).

---

You are reviewing a take-home submission for a **research position** (Latent Reasoning track,
task **LR-1: MeanFlow**). Be a rigorous, skeptical reviewer — your job is to find everything that
would make a strong research reviewer unimpressed, and to verify the work is actually correct.

**The task being graded:** Port **MeanFlow** (Geng et al., 2025, *Mean Flows for One-step Generative
Modeling*, `arXiv:2505.13447`) — a one-step image generator — from the official JAX/TPU code to a
clean **PyTorch/GPU** training loop. Prove it works by **overfitting 3 images** so that one-step
(1-NFE) generation reproduces them. Deliver a self-contained `report.html` plus a runnable repo.
The graders explicitly weight: (1) **correctness** of the MeanFlow objective and the one-step
sampler, (2) **quality of the explanation of problems faced** and the debugging/reasoning, and
(3) **code clarity and reproducibility**. (Note: the author had no GPU, so the verified runs are
on CPU at small scale; `train_imagenette.py` is the GPU path. Judge accordingly but flag if the
CPU framing undersells or overclaims anything.)

## Do this, in order

1. **Read** `README.md`, then `model.py`, `meanflow.py`, `train_overfit.py`, `utils.py`,
   `data.py`, `train_imagenette.py`, and skim `report.html` (open it in a browser if possible).

2. **Verify the math is implemented correctly** — this is the crux of the grade:
   - Interpolant `z_t = (1-t)x0 + t·eps`, conditional velocity `v = eps - x0`.
   - The MeanFlow identity `u(z,r,t) = v - (t-r)·d/dt u`, where `d/dt u = ∂_t u + (∂_z u)·v`
     is computed as a **JVP** with tangent `(v, 0, 1)` on inputs `(z, r, t)`.
   - **Stop-gradient** on the target; gradients only through the prediction.
   - One-step sampler `x0 = eps - u(eps, r=0, t=1)`.
   Confirm `meanflow.py` matches this and matches the paper's algorithm. Flag any sign error,
   wrong tangent, missing stop-grad, or mismatch with the paper.

3. **Run the correctness test:** `python test_meanflow.py` (CPU, ~20s). Confirm all 5 checks pass
   (the key one is JVP vs finite-difference). If anything fails, report it.

4. **Reproduce a quick result** (optional, ~1 min CPU):
   `python train_overfit.py --coupling fixed --size 16 --max_steps 300 --budget 100000`
   then `python train_overfit.py --finalize_only --outdir runs/overfit`. Confirm the residual
   drops and `runs/overfit/repro_1step.png` resembles `targets.png`.

5. **Critically assess** each area and report findings (don't rewrite code unless asked):
   - **Correctness:** any bug in the objective, JVP, sampler, EMA, checkpointing, or shapes?
   - **Code quality:** naming, structure, dead code, comments, edge cases, idiomatic PyTorch.
   - **Reproducibility:** do the README commands work? Is `requirements.txt` complete/pinned enough?
   - **Report:** are the explanations *accurate* (not just fluent)? Is the debugging narrative
     credible and specific? Flag any overclaim, hand-wave, or unsupported statement.
   - **Methodology judgment calls** — push hard on these:
     - Is **fixed coupling** a legitimate overfit-3 correctness test, or a way to dodge the harder
       random-coupling case? Is the author's justification sound?
     - Are the stability fixes (gap curriculum, high `p_eq`, adaptive weight) **principled** or ad-hoc?
     - Is the model size / resolution adequate to demonstrate the claim?
     - Does the author actually *understand* the JVP / why the bootstrap is unstable, or just
       pattern-match?

## Output
A prioritized review with sections: **(A) Correctness issues**, **(B) Code-quality issues**,
**(C) Report / clarity issues**, **(D) Concrete improvements (ranked)**, and **(E) Overall**: would
this submission land the research position? What are the 3 highest-leverage things to fix first?
Quote file:line where relevant. Be specific and honest — false praise is unhelpful.
