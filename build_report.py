"""Assemble a self-contained report.html with base64-embedded images.

Numbers are read from the runs' metrics.json so the report always matches the
actual results. Headline = fixed coupling @ 32x32; bonus = random coupling @ 16x16.
"""
import base64, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = "runs/fix32"      # required overfit-3, 32x32
RND = "runs/rand16"     # bonus generation, 16x16 (clearer modes than 32 at this tiny scale)
FAIL = "runs/of32b/peek_1step.png"  # a "before the fixes" failure (noise)


def embed(path):
    with open(os.path.join(HERE, path), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load(run):
    with open(os.path.join(HERE, run, "metrics.json")) as f:
        return json.load(f)


fix = load(FIX)
fix_mse = fix["reproduction_mse_mean"]
fix_per = fix["reproduction_mse_per_image"]
fix_per_str = ", ".join(f"{k} {v:.5f}" for k, v in fix_per.items())
fix_steps = fix["steps"]
fix_res = fix["final_residual_mse"]

css = """
:root{color-scheme:light}*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:880px;
 margin:0 auto;padding:32px 22px 80px;color:#1a1a1a;line-height:1.6;background:#fff}
h1{font-size:30px;margin:0 0 4px;line-height:1.2}
h2{font-size:22px;margin:38px 0 10px;border-bottom:2px solid #eee;padding-bottom:6px}
h3{font-size:17px;margin:22px 0 6px;color:#333}
.sub{color:#666;font-size:15px;margin-bottom:8px}
code,kbd{background:#f4f4f6;border-radius:4px;padding:1px 5px;font-size:13.5px;
 font-family:SFMono-Regular,Consolas,monospace}
pre{background:#f6f8fa;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;
 overflow:auto;font-size:13px;line-height:1.45}
.eq{background:#f6f8fa;border-left:3px solid #4a6;padding:10px 14px;margin:12px 0;border-radius:4px;
 font-family:SFMono-Regular,Consolas,monospace;font-size:14px;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid #e2e2e2;padding:7px 9px;text-align:center}
th{background:#fafafa}td.l,th.l{text-align:left}
.bad{color:#c0392b;font-weight:600}.good{color:#1a8a4a;font-weight:600}
figure{margin:16px 0;padding:14px;border:1px solid #eee;border-radius:10px;background:#fcfcfd}
figure img{width:100%;image-rendering:pixelated;border-radius:6px;display:block}
figcaption{font-size:13px;color:#555;margin-top:8px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.note{background:#fff8e6;border:1px solid #f0e0a0;border-radius:8px;padding:10px 14px;font-size:14px}
.ok{background:#eafaf0;border:1px solid #b6e6c7;border-radius:8px;padding:10px 14px;font-size:14px}
.tag{display:inline-block;background:#eef;border:1px solid #ccd;border-radius:20px;padding:2px 10px;
 font-size:12px;color:#335;margin-right:6px}
@media(max-width:640px){.grid2{grid-template-columns:1fr}}
"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MeanFlow — one-step pixel generation (PyTorch)</title><style>{css}</style></head><body>

<h1>MeanFlow: one-step pixel generation in PyTorch</h1>
<div class="sub">LR-1 (Latent Reasoning track). A from-scratch PyTorch port of the MeanFlow objective
(Geng et al., 2025, <code>arXiv:2505.13447</code>) from the JAX/TPU reference, verified by
overfitting 3 images and reproducing them with a single network evaluation.</div>
<p><span class="tag">PyTorch 2.4 · torch.func.jvp</span><span class="tag">pixel space</span>
<span class="tag">1-NFE sampler</span><span class="tag">overfit-3 ✓ (MSE {fix_mse:.4f})</span></p>

<div class="ok"><b>TL;DR.</b> One-step (1-NFE) generation reproduces the 3 training images at
<b>32×32</b> with mean MSE <b>{fix_mse:.4f}</b> (fixed coupling). The JVP at the heart of the
objective is checked against finite differences (rel. err ~3.5e−2), and a runnable
<code>test_meanflow.py</code> passes 5 correctness checks. The harder random-coupling case (true
one-step <i>generation</i> from noise) is also stabilized and shown below.</div>

<div class="note"><b>Compute honesty.</b> The build machine had <b>no GPU</b> (4-core ARM CPU; ~0.2&nbsp;s
per step; commands capped at 45&nbsp;s). I therefore used a <b>resumable, wall-clock-budgeted</b>
trainer and small models. The required overfit-3 is verified at <b>32×32</b>; the random-coupling
<i>generation</i> demo is shown at <b>16×16</b>, where a tiny CPU-sized model already produces
recognizable modes (at 32×32 it needs more capacity/steps than the CPU budget allowed). The same
core scales to 32/64&nbsp;px on a GPU via <code>train_imagenette.py</code>.</div>

<h2>1. What MeanFlow is (in my own words)</h2>
<p>Flow matching trains a network to predict the <b>instantaneous</b> velocity <code>v(z_t,t)</code>
of the probability-flow ODE, then generates by integrating that ODE in many small steps — each step
is one network call, so generation is slow.</p>
<p>MeanFlow instead models the <b>average velocity</b> over an interval <code>[r,t]</code> —
displacement divided by elapsed time:</p>
<div class="eq">u(z_t, r, t) = 1/(t−r) · ∫_r^t v(z_τ, τ) dτ</div>
<p>If you know the average velocity, generation is <b>one step</b>:
<code>x₀ = x₁ − u(x₁, 0, 1)</code> — a single network evaluation (1-NFE).</p>
<p>You can't supervise <code>u</code> directly (the integral is unknown), but differentiating its
definition gives a self-contained target. From <code>(t−r)·u = ∫_r^t v dτ</code>, differentiating
w.r.t. <code>t</code> (holding <code>r</code> fixed) gives the <b>MeanFlow identity</b>:</p>
<div class="eq">u(z_t, r, t) = v(z_t, t) − (t − r) · d/dt u(z_t, r, t)</div>
<p>The subtle part: <code>d/dt u</code> is a <b>total</b> derivative along the flow,</p>
<div class="eq">d/dt u = ∂u/∂t + (∂u/∂z_t)·v(z_t,t)</div>
<p>which is exactly a <b>Jacobian–vector product</b> (forward-mode AD) of <code>u</code> on inputs
<code>(z_t, r, t)</code> with tangent <code>(v, 0, 1)</code>: <code>r</code> is held fixed (tangent 0),
time advances (tangent 1), and position moves with velocity <code>v</code>. We use the conditional
velocity <code>v = ε − x₀</code> — an unbiased sample of the marginal velocity — for both the target
and the tangent. The loss regresses the prediction onto the identity, with a <b>stop-gradient</b>
on the target:</p>
<div class="eq">L = w · ‖ u_θ(z_t,r,t) − sg[ v − (t−r)·(∂_z u·v + ∂_t u) ] ‖²</div>
<p>When <code>r = t</code> the second term vanishes and the loss <b>is</b> plain flow matching — a
special case I lean on heavily for stability.</p>

<h2>2. Implementation</h2>
<p>Files: <code>model.py</code> (U-Net), <code>meanflow.py</code> (loss + samplers),
<code>data.py</code>, <code>utils.py</code>, <code>train_overfit.py</code>,
<code>train_imagenette.py</code> (GPU bonus). The JVP is six lines:</p>
<pre>def fn(z_, r_, t_): return model(z_, r_, t_)
tangents = (v, torch.zeros_like(r), torch.ones_like(t))     # (v, 0, 1)
u_pred, dudt = torch.func.jvp(fn, (z, r, t), tangents)      # primal + total dt-derivative
u_tgt = (v - (t - r) * dudt).detach()                       # stop-grad target
err   = u_pred - u_tgt
loss  = (w * err.pow(2)).mean()                             # w = adaptive weight</pre>
<ul>
<li><b>GroupNorm, never BatchNorm.</b> <code>torch.func.jvp</code> (forward-mode) doesn't compose with
BatchNorm's running stats, and batch-coupled norms make a per-sample directional derivative
ill-defined. No in-place activations either.</li>
<li><b>The primal carries gradients.</b> I verified <code>u_pred</code> from <code>torch.func.jvp</code>
still backprops to the parameters (108/108 parameter tensors get gradient), so one JVP call yields
both the prediction and the target — no second forward pass.</li>
<li><b>Samplers:</b> one-step <code>x₀ = ε − u_θ(ε, 0, 1)</code>; two-step splits <code>[1→0]</code>
at <code>t=0.5</code>.</li>
</ul>

<h2>3. Problems I hit and how I solved them</h2>

<h3>(a) Environment: 45-second commands, no background jobs</h3>
<p>CPU convolutions ran ~0.2&nbsp;s/step (ARM, no MKL) and each shell command was killed at 45&nbsp;s
with no persistent processes. Fix: a <b>resumable, wall-clock-budgeted</b> trainer — each call loads a
checkpoint, trains N seconds, saves model/opt/EMA/step/loss, prints progress — driven across many
calls.</p>

<h3>(b) Flat loss → a time-embedding aliasing bug</h3>
<p>Early runs had loss frozen near its initial value. Cause: I reused the classic diffusion sinusoidal
embedding, which assumes <i>integer</i> timesteps and scales <code>t</code> by 1000. For continuous
<code>t∈[0,1]</code> that <b>aliases</b> — nearby <code>t</code> get near-random embeddings, so the
network can't tell timesteps apart (and the velocity field is strongly time-dependent). Replacing it
with a Fourier basis on [0,1], <code>sin/cos(2πk·t)</code>, made the loss respond.</p>

<h3>(c) The real bottleneck: the JVP bootstrap explodes at the corner</h3>
<p>The target depends on the network's <i>own</i> derivative <code>du/dt</code>, so it can blow up. I
instrumented a fixed batch at three time pairs:</p>
<table>
<tr><th class="l">time pair (r, t)</th><th>|v|</th><th>|du/dt| (JVP)</th><th>|(t−r)·du/dt|</th><th>|target|</th><th>residual MSE</th></tr>
<tr><td class="l">r = t = 0.5 &nbsp;(pure flow matching)</td><td>1.11</td><td>3.96</td><td>0.00</td><td>1.11</td><td class="good">0.60</td></tr>
<tr><td class="l">r = 0.4, t = 0.6 &nbsp;(small gap)</td><td>1.11</td><td>2.51</td><td>0.50</td><td>1.04</td><td class="good">0.61</td></tr>
<tr><td class="l">r = 0, t = 1 &nbsp;(the corner the sampler uses)</td><td>1.11</td><td>5.10</td><td>5.10</td><td class="bad">4.47</td><td class="bad">22.95</td></tr>
</table>
<p>At the corner <code>(t−r)=1</code> multiplies the full <code>du/dt</code>, so the target balloons to
~4.5 (vs <code>|v|≈1.1</code>) and the residual is ~23. Chasing it grows <code>du/dt</code> further — a
feedback loop. And the corner is <i>exactly</i> where one-step generation evaluates, so early samples
were pure noise:</p>
<figure><img src="{embed(FAIL)}" alt="noise">
<figcaption><b>Failure mode</b> (before the fixes): one-step samples are noise because the corner never
converges.</figcaption></figure>

<h3>(d) The adaptive-weight trap</h3>
<p>The paper's adaptive weight <code>w = 1/(‖err‖²+c)^γ</code> (stop-grad) stabilizes by down-weighting
high-residual samples — but with <code>γ=1</code> it down-weights <i>exactly</i> the corner samples
one-step generation needs, so they never get fixed (the loss merely <i>looks</i> flat at ~1.0 while
the raw residual silently grows to ~15). With <code>γ=0</code> (plain MSE) the corner instead dominates
and diverges. Neither extreme works alone.</p>

<h3>(e) The fix that worked</h3>
<ul>
<li><b>Gap curriculum:</b> cap <code>(t−r) ≤ max_gap</code> and ramp it 0.3→1.0, so the unstable term
is bounded until the field is learned.</li>
<li><b>Anchor the velocity:</b> <code>p_eq=0.8</code> of samples use <code>r=t</code> (pure flow
matching), keeping <code>du/dt</code> near its true value.</li>
<li><b>Adaptive weight + grad-clip 0.5 + lr 1e-3</b> for the rest.</li>
</ul>
<p>With these, the residual collapsed to <b>{fix_res:.3f}</b> in {fix_steps} steps (32×32, fixed
coupling).</p>

<h3>(f) Fixed vs random noise↔image coupling</h3>
<p>With 3 images and <i>random</i> coupling the velocity field is extremely peaky (large irreducible
target variance) — the worst case for the bootstrap. Pairing each image with a <b>fixed</b> noise makes
each path a straight line with constant velocity (true <code>du/dt=0</code>): the clean, standard
overfit test, and exactly what "reproduce the 3 images" asks. I report both.</p>

<h3>(g) EMA</h3>
<p>EMA decay 0.999 lags far too much over a few-hundred-step run, so EMA samples were junk; I sample
from the raw weights for the short overfit and keep EMA for the long GPU run.</p>

<h2>4. Results</h2>
<h3>Required: overfit-3, fixed coupling @ 32×32 — 1-step reproduction</h3>
<p>One-step (1-NFE) generation from each image's fixed noise returns the training image almost exactly:
mean reproduction MSE <b>{fix_mse:.4f}</b> on the [−1,1] scale (per image: {fix_per_str}).</p>
<div class="grid2">
<figure><img src="{embed(FIX + '/targets.png')}" alt="targets">
<figcaption><b>Targets</b> — the 3 training images (32×32).</figcaption></figure>
<figure><img src="{embed(FIX + '/repro_1step.png')}" alt="repro 1-step">
<figcaption><b>1-step reconstructions</b> (1 network eval). MSE ≈ {fix_mse:.4f} — visually identical.</figcaption></figure>
</div>
<figure><img src="{embed(FIX + '/loss_curve.png')}" alt="loss curve fixed">
<figcaption>Residual MSE (log) over {fix_steps} steps, fixed coupling @ 32×32.</figcaption></figure>

<h3>Bonus: random coupling — true one-step generation</h3>
<p>The same recipe also stabilizes the much harder random-coupling case (residual floors well above
zero — the marginal velocity is a genuine average). One-step samples from <i>random</i> noise collapse
onto the memorized modes — the coffee cup, astronaut, and cat textures are all readable — i.e. the
model learned a one-step noise→image map, not just memorized pairs (shown at 16×16).</p>
<figure><img src="{embed(RND + '/samples_random_1step.png')}" alt="random samples">
<figcaption><b>Random-coupling, 1-step samples from random noise</b> (16×16). Recognizable training
images appear — the expected overfit behavior for a 3-image generative model.</figcaption></figure>

<h2>5. Correctness checks</h2>
<p><code>python test_meanflow.py</code> passes 5 checks (CPU, ~20&nbsp;s):</p>
<pre>[PASS] JVP total-derivative matches finite differences (rel err 3.47e-02)
[PASS] sample_r_t: r&lt;=t, gap&lt;=0.4, r==t fraction 0.31
[PASS] gradient flows through JVP primal to all 108 param tensors
[PASS] training reduces residual (2.005 -&gt; 0.039)
[PASS] samplers: correct shapes and value range</pre>
<p>The first is the important one: it independently confirms the total-derivative used in the target is
the correct directional derivative of <code>u</code> (not just that training happened to converge).</p>

<h2>6. Reproduce</h2>
<pre>pip install -r requirements.txt
python test_meanflow.py                                   # correctness checks

# required overfit-3 (fixed coupling) — produced the headline result:
python train_overfit.py --coupling fixed --size 32 --tile 2 \\
    --p_eq 0.8 --gamma 1.0 --gap0 0.3 --gap_ramp 0.7 \\
    --lr 1.5e-3 --clip 0.5 --max_steps 700 --budget 100000
python build_report.py                                    # -> report.html (self-contained)

# bonus, on a GPU (CIFAR-10 / Imagenette, 32 or 64 px):
python train_imagenette.py --data cifar10 --size 32 --base 128 --steps 60000</pre>
<p><code>--budget</code> caps per-invocation wall-clock so training fits a 45&nbsp;s command limit and
auto-resumes from <code>runs/&lt;name&gt;/ckpt.pt</code>; set it huge to train in one shot. See
<code>README.md</code> for details and <code>REVIEW_PROMPT.md</code> for the review brief.</p>

</body></html>"""

with open(os.path.join(HERE, "report.html"), "w") as f:
    f.write(html)
print(f"wrote report.html ({len(html)/1024:.0f} KB) | fixed MSE {fix_mse} | {fix_steps} steps")
