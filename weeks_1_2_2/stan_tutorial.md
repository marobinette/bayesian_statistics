# A from-scratch Stan tutorial, built around your notebook

This gets you ready to work through `MVP_Stan_notebook.ipynb` with a real understanding of what every line is doing, not just what to run. It assumes you can program and handle basic math, but it assumes you have never done Bayesian statistics. Everything is anchored to the exact model in your notebook (the "globe toss") and picks up from the `bernoulli` model you already ran on the command line.

By the end you should be able to look at the notebook's Stan model and say, out loud, what each block means, what sampling produced, and how to check the answer.

---

## 0. Where you are right now

You have done two things already:

1. Installed CmdStan (the command-line version of Stan) on your Mac and confirmed it works by running the `bernoulli` example.
2. Seen a `stansummary` table with a parameter `theta` estimated at about 0.26.

Your notebook does the *same kind of thing* through a different door. Instead of driving Stan from the terminal, it drives Stan from Python using a package called **CmdStanPy**. The model in the notebook (the globe toss) is mathematically almost identical to the `bernoulli` model you already ran. So you are not starting over. You are re-meeting something familiar through Python, and this time understanding the statistics underneath it.

Two names to keep straight:

- **CmdStan**: the actual Stan engine. Compiles your model to C++, runs the sampler. This is what you installed.
- **CmdStanPy**: a thin Python wrapper that writes your model to a file, calls CmdStan to compile and run it, and hands the results back as Python objects. This is what the notebook uses.

CmdStanPy does not replace CmdStan. It calls it. (More on wiring your existing CmdStan install into CmdStanPy in section 6.)

---

## 1. The one idea behind everything: Bayesian inference

Forget Stan for a moment. Here is the entire idea.

You have some unknown quantity you care about. In the globe toss, you toss an inflatable globe, catch it, and record whether your right index finger landed on **water** or **land**. Do this `N` times, count `K` waters. You want to know: **what fraction of the globe's surface is water?** Call that unknown fraction `rho` (the notebook calls it `rho`). It is a number between 0 and 1.

You will never observe `rho` directly. You only observe tosses. Bayesian inference is a recipe for going from "I observed `K` waters out of `N` tosses" to "here is what I now believe about `rho`, expressed as a probability distribution."

That last phrase is the whole point. The answer is **not** a single number. The answer is a distribution over all the values `rho` could be, saying how plausible each one is given your data. A single number (like "0.45") is just a summary of that distribution.

### The three pieces

Bayesian inference combines exactly three things.

**1. The prior: `p(rho)`.** What you believe about `rho` *before* seeing any tosses. Maybe you think any fraction is equally likely. Maybe you think it is probably near 0.5 but you're not sure. You encode that belief as a probability distribution over `rho`.

In your notebook the prior is `beta(1.5, 1.5)`. The Beta distribution is the natural choice for an unknown probability because it lives entirely on the interval [0, 1], which is exactly the range `rho` can take. `beta(1.5, 1.5)` is a gentle hump centered at 0.5: it says "probably somewhere in the middle, but I'm barely committed to that." A `beta(1, 1)` would be perfectly flat, meaning "I have no idea, all fractions equally likely" (that flat one is what the `bernoulli` example used implicitly, which is why its estimate landed right on the raw data proportion).

**2. The likelihood: `p(data | rho)`.** If `rho` had some *specific* value, how probable would the data you actually saw be? For the globe toss, each toss is water with probability `rho`, independently, so the count of waters in `N` tosses follows a **binomial** distribution:

```
K ~ binomial(N, rho)
```

Read that as: "the observed count `K` is a draw from a binomial distribution with `N` trials and success probability `rho`." For a given `rho`, the binomial tells you how likely `K` waters was.

**3. The posterior: `p(rho | data)`.** What you believe about `rho` *after* folding in the data. This is the answer you want. Bayes' rule says how to get it:

```
posterior(rho | data)  =  likelihood(data | rho) * prior(rho)  /  p(data)
```

or in the form people actually use:

```
posterior(rho | data)  is proportional to  likelihood(data | rho) * prior(rho)
```

In words: **your updated belief is the prior belief reweighted by how well each value of `rho` explains the data.** Values of `rho` that make the observed data likely get boosted; values that make it unlikely get suppressed. Data pulls you away from the prior; more data pulls harder.

The `p(data)` in the denominator (called the "evidence" or "marginal likelihood") is just a normalizing constant that makes the posterior integrate to 1. Hold onto it. It is the villain of section 3.

### Working the globe toss by hand

Here is the payoff of picking this example: for the Beta prior + binomial likelihood combination, the posterior has a known closed form. You can compute the exact answer with a pen, and then check that Stan recovers it. That is the best possible way to build trust in a new tool.

The rule (this is a standard result, the Beta is "conjugate" to the binomial):

```
prior beta(a, b)  +  observe K successes in N trials
   =>  posterior beta(a + K, b + N - K)
```

Plug in the notebook's numbers. The prior is `beta(1.5, 1.5)`, so `a = 1.5`, `b = 1.5`. The notebook's data cell uses `N = 7`, `K = 3`. So `N - K = 4`. The posterior is:

```
beta(1.5 + 3, 1.5 + 4)  =  beta(4.5, 5.5)
```

The mean of a `beta(a, b)` is `a / (a + b)`, so the posterior mean is:

```
4.5 / (4.5 + 5.5)  =  4.5 / 10  =  0.45
```

Now notice what happened. The raw data proportion is `3/7 = 0.4286`. The prior was centered at 0.5. The posterior mean, 0.45, sits between them, pulled from the data proportion a little toward the prior's center. That pull is the prior doing its job. Because the prior is weak (it only carries the weight of about 3 pseudo-observations), the pull is small and the data dominates. If you had tossed the globe 700 times instead of 7, the prior's influence would be almost invisible and the posterior mean would sit right on top of the data proportion.

This is the mechanical heart of Bayesian updating, and you just did it by hand. Everything Stan does is a way to get this same posterior when you *can't* do it by hand.

### Compare to the run you already did

Your `bernoulli` run was the same structure with different numbers: a flat `beta(1, 1)` prior and data of 2 successes in 10. Posterior `beta(1 + 2, 1 + 8) = beta(3, 9)`, mean `3/12 = 0.25`. Your `stansummary` showed `theta` at 0.26. That 0.26 was Stan's sampled approximation of the exact 0.25 you can now derive yourself. Same story, and now you can see the machinery.

---

## 2. Why we can't always do it by hand (enter MCMC)

The globe toss is a trap in one nice way: it is solvable on paper. Almost nothing you will actually care about is.

The problem is that denominator, `p(data)`. To compute it you have to integrate the numerator over *every possible value of the parameters*:

```
p(data) = integral over all rho of [ likelihood(data | rho) * prior(rho) ] d rho
```

For one parameter on [0, 1] with a conjugate setup, that integral has a formula. For a model with 200 parameters, coupled in nonlinear ways (say, a contagion model over a network with node-level and group-level effects), that integral is a 200-dimensional monster with no closed form and no hope of numerical quadrature. You cannot compute the normalizing constant, so you cannot write down the posterior.

**Markov chain Monte Carlo (MCMC)** is the escape. Here is the key insight: you do not actually need the formula for the posterior. You need to be able to *answer questions* about it, like "what's the mean of `rho`?" or "what's the probability `rho > 0.5`?". And you can answer any such question if you have a big pile of **samples drawn from the posterior**, even without ever knowing its formula.

If I hand you 1000 values of `rho` drawn in proportion to how plausible the posterior says they are, then:

- the average of those 1000 values estimates the posterior mean,
- the fraction of them above 0.5 estimates `P(rho > 0.5)`,
- their 5th and 95th percentiles give you a 90% credible interval,

and so on. Samples turn hard integrals into easy averages. That is the entire trick.

The remarkable part: MCMC can generate samples from the posterior using **only the numerator**, `likelihood * prior`, which you *can* always compute. It never needs the intractable denominator, because the denominator is a constant that does not depend on `rho`, and the sampler only ever cares about the *ratio* of plausibilities between two candidate values, where the constant cancels.

Stan's specific sampler is **HMC/NUTS** (Hamiltonian Monte Carlo, with the No-U-Turn Sampler on top). You do not need its internals yet. The one-sentence version: it treats the posterior as a landscape, rolls a frictionless particle around it using gradient information, and records where the particle spends time. Regions the particle visits often are high-posterior regions. The recorded positions are your samples. It uses gradients (derivatives of `log(likelihood * prior)`), which is why Stan needs your model written in a differentiable way, and why it is so much more efficient than older samplers on hard problems.

When you saw `Warmup took 0.008 seconds / Sampling took 0.012 seconds` and a column of numbers, that was this process: a particle explored the `beta(3, 9)` landscape and left behind 1000 positions, whose average was 0.26.

---

## 3. Anatomy of a Stan program

A Stan model is a text file (`.stan`) built from labeled **blocks**. You saw three of them in your `bernoulli` model and you will see the same three in the notebook. Here is the notebook's globe-toss model with every line explained.

```stan
data {
  int<lower=0> N;
  int<lower=0, upper=N> K;
}
parameters {
  real<lower=0, upper=1> rho;
}
model {
  rho ~ beta(1.5, 1.5);
  K ~ binomial(N, rho);
}
```

### The `data` block

```stan
data {
  int<lower=0> N;
  int<lower=0, upper=N> K;
}
```

This declares the things you will supply from outside, the observed quantities and constants. Nothing here is estimated. `N` is an integer that cannot be negative (`<lower=0>`). `K` is an integer between 0 and `N` (`<lower=0, upper=N>`). Those bounds are **constraints Stan will check for you**: if you accidentally pass `K = 9, N = 7`, Stan errors out instead of silently producing garbage. Declaring types and bounds here is not bureaucracy, it is a correctness net.

You provide these values at run time. In the notebook that happens in the sampling call: `data={"N": 7, "K": 3}`. The keys match the names in this block exactly.

### The `parameters` block

```stan
parameters {
  real<lower=0, upper=1> rho;
}
```

This declares the unknowns you want to infer, the things the posterior is *over*. Here there is exactly one: `rho`, a real number constrained to [0, 1] because it is a proportion. That constraint matters more than it looks: Stan's sampler works in an unconstrained space internally and transforms back and forth, so `<lower=0, upper=1>` guarantees every sample it proposes is a valid proportion. You never get a nonsensical `rho = 1.3`.

Whatever you put in this block is what shows up in your results table as a quantity to be estimated. `rho` here plays the exact role `theta` played in your `bernoulli` run.

### The `model` block

```stan
model {
  rho ~ beta(1.5, 1.5);
  K ~ binomial(N, rho);
}
```

This is where you write the prior and the likelihood, the two ingredients from section 1. Read the `~` as "is distributed as."

- `rho ~ beta(1.5, 1.5);` is the **prior**. It says: before data, `rho` is distributed as a `beta(1.5, 1.5)`.
- `K ~ binomial(N, rho);` is the **likelihood**. It says: the observed count `K` is distributed binomially with `N` trials and success probability `rho`.

Here is the subtle but important thing about what these lines actually do. They do not "draw" anything. Each `~` statement *adds a term to the log of `likelihood * prior`* that Stan is accumulating internally. When the sampler needs to know "how plausible is this candidate `rho`?", it runs the `model` block to add up: the log-prior density of that `rho`, plus the log-likelihood of the data under that `rho`. That running total is the (unnormalized) log-posterior the sampler climbs around on. So the `model` block is really a function that scores a candidate parameter value. Writing it with `~` just hides the bookkeeping.

Notice what is *absent*: there is no line computing the posterior, no normalizing constant, no `beta(4.5, 5.5)`. You never tell Stan the answer. You only tell it the prior and the likelihood, the two pieces you can always write down, and it works out the posterior by sampling. The fact that this particular model happens to have a clean conjugate answer is invisible to Stan and irrelevant to it. It runs HMC either way. Conjugacy is *our* luxury for checking the result, not something Stan uses.

### Blocks you will meet later

Real models use more blocks. You do not need them for the notebook, but so the names are not a surprise:

- `transformed data`: precompute constants from the data once, before sampling.
- `transformed parameters`: define quantities that are deterministic functions of your parameters (e.g. reparameterizations).
- `generated quantities`: compute things *after* each sample, like predictions for new data or posterior predictive checks. This runs once per draw and does not affect the fit.
- `functions`: your own reusable functions.

---

## 4. The Bayesian workflow, and where the notebook sits in it

The honest version of doing this well is a loop, not a single run:

1. Write down a model (prior + likelihood).
2. Compile it.
3. Sample from the posterior.
4. Check that the sampling itself was trustworthy (diagnostics: R-hat, divergences, ESS).
5. Check that the model makes sense against reality (posterior predictive checks).
6. Revise and repeat.

Your notebook is a deliberately minimal pass through steps 1 to 3, plus a bare "did it run at all" check. It is a *screen test*, not a full analysis. Knowing that is the point: once you understand what it is skipping (steps 4 to 6), you know what you would add for real work. Section 7 shows the first thing to add.

---

## 5. Reading the notebook cell by cell

Now the notebook itself. Four cells.

### Cell 1: install and get CmdStan

```python
%pip install -q cmdstanpy==1.3.0
...
CMDSTAN_VERSION = "2.39.0"
...
cmdstanpy.set_cmdstan_path(str(cmdstan_dir))
```

This installs the CmdStanPy Python package, then downloads a **prebuilt** CmdStan (the engine) specifically packaged for Colab and points CmdStanPy at it with `set_cmdstan_path`. The prebuilt download exists because compiling CmdStan from scratch inside a fresh Colab runtime is slow, and Colab wipes the runtime regularly. This is the cell the notebook's own header warns you about: on Colab you pay this setup cost every time the runtime disconnects.

On your Mac this whole cell is unnecessary, because you already built CmdStan once and it persists. See section 6.

### Cell 2: write the model to a file

```python
stan_code = r"""
data { ... }
parameters { ... }
model { ... }
"""
stan_file = Path("/content/globe_toss.stan").write_text(stan_code)
```

This just writes the Stan program (the one you dissected in section 3) to a file called `globe_toss.stan`. Stan models live in their own files; this cell is the Python-friendly way to create one inline. Nothing statistical happens here, it is file I/O. The `r"""..."""` is a raw string so backslashes in Stan code would not be mangled by Python.

### Cell 3: compile and sample

```python
model = CmdStanModel(stan_file=str(stan_file))

fit = model.sample(
    data={"N": 7, "K": 3},
    chains=2,
    parallel_chains=2,
    iter_warmup=250,
    iter_sampling=250,
    seed=6300,
    show_progress=False,
)
print("Stan screen test passed.")
```

Two distinct things happen on these lines.

`CmdStanModel(stan_file=...)` **compiles** your model. Under the hood CmdStanPy calls CmdStan to translate the `.stan` file to C++ and compile it to a native executable, exactly the three-stage "Translating / Compiling / Linking" sequence you watched scroll by when you ran `make examples/bernoulli/bernoulli`. The first time is slow (the notebook header quotes about 50 seconds). After that the compiled executable is cached and reused, so re-running is instant unless you change the model.

`model.sample(...)` **runs the sampler**. This is the step that actually produces posterior draws, the equivalent of your `./bernoulli sample ...` command. The arguments:

- `data={"N": 7, "K": 3}`: supplies the `data` block. Keys match the Stan names.
- `chains=2`: run 2 independent samplers from different random starts. Running several and comparing them is how you detect trouble (see R-hat below). Your `bernoulli` run used only 1 chain, which is why I flagged its convergence check as weak.
- `parallel_chains=2`: run those 2 chains at the same time on separate cores. This is exactly the multi-core parallelism from your very first question, showing up in Python. On your 8-core i9 you could run 4 or 8 chains in parallel comfortably.
- `iter_warmup=250`: 250 tuning iterations where HMC adapts its step size and geometry. These are discarded, not kept.
- `iter_sampling=250`: 250 kept draws per chain. With 2 chains that is 500 posterior samples total.
- `seed=6300`: fixes the random number generator so the run is reproducible. Same seed, same draws. Always set this.
- `show_progress=False`: hides the progress bars.

The final `print` only runs if `model.sample` returned without throwing. That is the whole "screen test": it proves compilation and sampling both work in this environment. It does **not** check that the answer is any good. That is on you, next.

### What the notebook does not do

The notebook never looks at `fit`. It samples and immediately declares success. For real work the `fit` object is the entire point, so let's use it.

---

## 6. Running this on your Mac instead of Colab

You do not need Colab or the prebuilt download. You already have a working CmdStan. To use it from Python:

```bash
pip install cmdstanpy
```

Then in Python, point CmdStanPy at your existing install (the directory where you ran `make`):

```python
import cmdstanpy
cmdstanpy.set_cmdstan_path("/Users/marobinette/cmdstan")
print(cmdstanpy.cmdstan_path())
```

Alternatively set the environment variable `CMDSTAN=/Users/marobinette/cmdstan` before launching Python and CmdStanPy will find it automatically. Either way, `CmdStanModel(...)` will now reuse your existing compiled backend and cache compiled models on disk, so you get the persistence the notebook's header is nudging you toward. Everything in cells 2 and 3 then works unchanged; you just skip cell 1's download entirely.

---

## 7. Actually reading the results (the part the notebook skips)

Replace the bare `print` with a look at `fit`. Two views matter.

### The summary table

```python
print(fit.summary())
```

This is the CmdStanPy equivalent of the `stansummary output.csv` you already ran, and it produces the same columns. For the globe toss you are looking at the row for `rho`. Check three things, in order:

**1. Is the sampling trustworthy?** Look at `R_hat`. This compares your 2 chains: if independent chains that started in different places ended up exploring the same distribution, they agree, and `R_hat` is very close to 1.00. If it is above about 1.01, the chains disagree and you should not trust the numbers. This is *why you run more than one chain*, and why a single-chain `R_hat` (like your `bernoulli` run) is weak evidence. Also check that `divergent__` is 0. Divergences are the sampler signaling it hit geometry it could not handle; a nonzero count means the results may be biased. For this easy model both should be pristine.

**2. Is there enough information?** Look at `ESS_bulk` and `ESS_tail`, the effective sample sizes. MCMC draws are autocorrelated, so 500 draws are worth *fewer* than 500 independent draws. ESS estimates how many independent draws you effectively have. A few hundred is fine for a mean; you want more (thousands) if you care about extreme tail quantiles.

**3. Only now, the answer.** Look at `Mean` for `rho`. It should be about **0.45**, the number you computed by hand in section 1. Also look at the `5%` and `95%` columns: those bracket a 90% **credible interval**, the range the posterior puts 90% of its belief in. Unlike a frequentist confidence interval, you *can* say "there's a 90% probability `rho` is in here," because the posterior is a genuine distribution over `rho`.

If `Mean` is near 0.45 and `R_hat` is 1.00 with no divergences, you have not just run Stan, you have *confirmed it recovered the exact posterior you derived by hand*. That is the moment to trust the tool.

### The raw draws

```python
draws = fit.stan_variable("rho")   # a numpy array of every posterior draw of rho
print(draws.mean(), draws.std())
print((draws > 0.5).mean())        # posterior probability that rho > 0.5
```

`fit.stan_variable("rho")` hands you the actual samples as a NumPy array. This is the pile-of-samples from section 2, and now you can compute *any* posterior quantity as a plain array operation. `(draws > 0.5).mean()` is the posterior probability that water covers more than half the globe, computed as a one-liner. This is the concrete meaning of "samples turn integrals into averages." Try to answer that same question from a `stansummary` table alone and you will appreciate why you want the draws.

For a quick visual, a histogram of `draws` *is* a picture of the posterior. It should look like a hump peaked near 0.44, which is the `beta(4.5, 5.5)` density you derived.

---

## 8. Small exercises to make it stick

Do these in your Mac setup from section 6. Each one isolates a single concept.

1. **Watch the data dominate the prior.** Rerun with `data={"N": 70, "K": 30}` (same proportion, ten times the data). The posterior mean should move from 0.45 toward the data proportion 0.4286, because the fixed prior now carries relatively less weight. Confirm the exact target by hand: posterior `beta(1.5 + 30, 1.5 + 40) = beta(31.5, 41.5)`, mean `31.5 / 73 = 0.4315`. Watch the credible interval get narrower too: more data, less uncertainty.

2. **Watch the prior matter when data is scarce.** Go back to `N=7, K=3` but change the prior to `beta(20, 20)`, a strong belief that `rho` is near 0.5. The posterior mean should get yanked much closer to 0.5, because the prior now outweighs 7 tosses. Derive the target: `beta(23, 24)`, mean `23/47 = 0.489`. This is the clearest possible demonstration that priors are not decoration.

3. **Break it on purpose.** Pass `data={"N": 7, "K": 9}` and read the error. `K` violates the `<lower=0, upper=N>` constraint you declared in the `data` block. Seeing Stan catch this teaches you why those bounds are worth writing.

4. **Ask a posterior question the table can't.** Using `fit.stan_variable("rho")`, compute the posterior probability that `rho` is between 0.3 and 0.5. This forces you to use the draws directly, which is the skill that generalizes to every model you will ever fit.

---

## 9. Where this goes next

Everything above scales. The workflow (write prior + likelihood in blocks, compile, sample, check diagnostics, inspect draws) is identical whether you have one parameter or ten thousand. What changes as models get harder:

- **More parameters and structure.** The natural next step is a *hierarchical* model, where parameters themselves have priors governed by higher-level parameters (e.g. per-group effects drawn from a shared distribution). This is where Bayesian modeling earns its keep, and where partial pooling comes from.
- **Diagnostics start to bite.** On the globe toss, `R_hat` and divergences are always clean. On real models they are not, and reading them (plus reparameterizing to fix them, the "non-centered parameterization" is the classic move) becomes a core skill.
- **Model checking.** The `generated quantities` block lets you simulate data from your fitted model and compare it to reality, which is how you find out your model is wrong.

Given your work on contagion and network dynamics, the connection is direct: a spreading process with unknown transmission rates, group-level infection terms, and node heterogeneity is exactly a hierarchical Bayesian model. The globe toss is the same machinery with the complexity dialed to one. Get fully comfortable with this one parameter, diagnostics included, and the jump to a contagion likelihood is a change of `model` block, not a change of method.

For reference as you go: the CmdStanPy docs (the Python API you're using), the CmdStan guide (the engine you installed), and the Stan Functions Reference (every distribution and function available in the `model` block) are the three you will actually keep open.
