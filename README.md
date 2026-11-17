# Virtual Pleasure Experiment (VPE) 0.3.0

A sequence simulator for investigating a written sexual-technique guide across 1,000
synthetic adult women. Paste a guide, correct the parsed steps, then compare
protocols, remove steps, reorder them, and see which women gain and which lose.

**Read this before using any number it produces.** The women are generated. The
population's satisfaction *level* is calibrated to published survey marginals. Every
technique, sequence and comfort *mechanism* is a hypothetical assumption — no
published study estimates the effect of a specific cunnilingus technique, and this
project does not pretend otherwise. The tool explores what your assumptions imply.
It is not evidence that any technique works.

---

## 1. What changed in 0.3.0

0.2.0 was a demonstration that a hypothetical model could not tell you anything. It
had seven sliders, a free-text box the model ignored, and a single "assumed effect"
slider that did all the work. That is replaced.

| 0.2.0 | 0.3.0 |
| --- | --- |
| Seven protocol sliders | An **ordered sequence of technique steps** with durations and intensities |
| Free-text guide ignored by the model | **Rule-based parser** turns your guide into steps, which you then review and correct |
| One "assumed effect" slider | **Removed.** Effects arise from the sequence mechanisms and their interaction with each woman's profile |
| One aggregate outcome set | **Ablation, ordering, substitution and subgroup studies** built into every run |
| Couples | **1,000 individual women** with latent preference archetypes |
| Citations listed | **Evidence database**: six verified sources, each with what it does and does not support, plus an evidence level on every parameter |
| Null scenario arbitrary | **Calibrated** to published satisfaction marginals, with the deviation reported |

Nothing was mocked and no scientific feature was removed. The continuous
pleasure/arousal/comfort/satisfaction equations are still there — they are now
integrated over time instead of evaluated once.

---

## 2. How it works

### 2.1 Guide input (`guide_parser.py`)

Paste your guide. The parser splits it into blocks (numbered lines, bullets, blank
lines, or long sentences), then looks for technique phrases, duration words
("several minutes", "30 seconds") and intensity modifiers ("gentle", "firm").

It is a **keyword matcher, not a language model**. Three things it does that are
worth knowing:

- It **respects negation**: "work over the hood, not the glans" does not create a
  glans step, and "no direct clitoral contact yet" does not either.
- It **reports what it could not classify** instead of guessing, and those blocks are
  shown to you to add manually.
- It **flags ambiguity**: a bare "labia" could be outer or inner, so it says so.

You then edit every step's technique, minutes and intensity in an editable table.
That corrected table is the experiment's input. The review step is the honest
substitute for "the AI understood my guide" — it did not, and you are the authority
on what your guide means.

### 2.2 Population (`population.py`)

1,000 synthetic adult women (18+), generated from five latent **preference
archetypes** plus individual variation: prefers direct and firm; prefers indirect
and gentle; needs sustained rhythm; prefers broad diffuse stimulation; highly
sensitive. Allocation is **stratified**, so each population contains the archetype
proportions almost exactly rather than drifting with a multinomial draw.

Each woman carries: preferred intensity, preferred directness, intensity tolerance,
warm-up requirement (minutes), overstimulation threshold, rhythm dependence, comfort
sensitivity, communication openness, experience, obligation sensitivity.

The archetypes and weights are a modelling device. They are **not** prevalence
estimates, and no source claims 22% of women are one type.

### 2.3 Sequence model (`sequence_model.py`)

Arousal is integrated minute by minute against whichever technique is being applied.
Four mechanisms do the work:

1. **Fit** — a technique is pleasant in proportion to how close its intensity and
   directness are to that woman's preference, within her tolerance band.
2. **Warm-up** — direct clitoral contact before her warm-up requirement is met is
   penalised and accrues discomfort. This is the mechanism behind "do not open with
   suction on the glans".
3. **Overstimulation** — every technique loads an overstimulation channel; past her
   threshold, added intensity subtracts from arousal and adds discomfort. This is why
   "more" stops working.
4. **Rhythm** — sustained, unchanging stimulation accumulates a plateau bonus
   proportional to her rhythm dependence; switching technique resets it.

Outcomes are logistic functions of accumulated state, so they lie strictly in (0, 1)
and are unitless indices. Discomfort is measured as a **rate per minute** and
saturated, so a long well-paced protocol is not punished simply for lasting longer
than a short badly-sequenced one.

### 2.4 Calibration (`calibration.py`)

The published survey reports the distribution of satisfaction with receiving oral sex
across four ordered categories. Two parameters of the satisfaction equation (its
intercept and scale) were fitted so the model's **baseline scenario** reproduces those
proportions. Method: coarse grid search minimising the mean, across five seeds, of the
maximum absolute band deviation.

Result: **1.2–2.4 percentage points per band across five seeds at 1,000 women**
(2.41 at the default seed). The deviation grows with smaller populations — at 100
women sampling noise alone can reach 9 points, and the app reports the achieved
deviation for your run rather than hiding it.

What calibration buys: the null scenario is not arbitrary. What it does not buy: any
evidence about technique. A calibrated model can still be structurally wrong.

---

## 3. What you can investigate

| Question | Where |
| --- | --- |
| Which techniques contribute most to pleasure? | Ablation: remove each step, re-run |
| Does the order matter? | Sequence variants: as written, reversed, direct-first, warm-up removed, communication removed |
| What if a step is removed or replaced? | Ablation and substitution tables |
| Which combinations work? | Compare any two protocols |
| Which women is a technique unsuitable for? | Subgroup analysis by archetype and by profile variable |
| How much does the answer depend on assumptions? | Sensitivity: one-at-a-time perturbation of every coefficient |

In the worked example (a paced guide against an unstructured baseline), removing the
warm-up steps costs about 0.54 on the pleasure index and adds about 0.41 on
discomfort — the largest single effect in the whole run. That is what the model says
about *these assumptions*, and it is the kind of statement this tool can support.

---

## 4. Evidence position

Six peer-reviewed sources are used, all with citations verified against the published
record on 17 November 2026:

| Source | N | Used for |
| --- | --- | --- |
| Ogallar-Blanco, Lara-Moreno & Godoy-Izquierdo (2022), IJERPH 19(16):10232 | 1,080 | Calibration target: satisfaction marginals for receiving oral sex |
| Richters, de Visser, Rissel & Smith (2006), J Sex Res 43(3):217-226 | 19,307 | More practices, especially cunnilingus, associated with orgasm |
| Herbenick, Fu, Arter, Sanders & Dodge (2018), J Sex Marital Ther 44(2):201-212 | 1,055 | Preference heterogeneity is large |
| Frederick, St. John, Garcia & Lloyd (2018), Arch Sex Behav 47(1):273-288 | 52,588 | Practice heterogeneity at scale |
| Backstrom, Armstrong & Puentes (2012), J Sex Res 49(1):1-12 | 43 | Consent/negotiation as distinct from technique |
| Rosen et al. (2000), J Sex Marital Ther 26(2):191-208 | 259 | Arousal, comfort, satisfaction, orgasm are separable |

**What none of them does: estimate the effect of a technique, a pressure, a rhythm or
a sequence.** Every such parameter in this model is labelled level E, *unvalidated
model relationship*, and the app prints that label beside every result. The evidence
database in `vpe/evidence.py` states, for each study, what it supports and what it
does not.

---

## 5. Run locally

Python 3.11 or newer (tested on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

---

## 6. Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository. **The repository root must be the folder
   containing `app.py`.**
2. Go to <https://share.streamlit.io>, sign in with GitHub, **Create app**.
3. Repository `<you>/<repo>`, branch `main`, main file path `app.py`.
4. **Advanced settings** → Python **3.12**. No secrets needed.
5. Deploy. First build takes 2–5 minutes.
6. Press **Run experiment** to confirm the engine executes in the cloud.

Updating: push to `main`; Streamlit rebuilds automatically.

---

## 7. Project structure

```text
vpe/
├── app.py                  # dashboard, sections A-G
├── vpe/
│   ├── evidence.py         # research database + evidence levels
│   ├── techniques.py       # technique catalogue and its parameters
│   ├── guide_parser.py     # rule-based guide -> steps
│   ├── population.py       # 1,000 women, latent archetypes
│   ├── protocol.py         # Step / Protocol, baseline convention
│   ├── sequence_model.py   # minute-by-minute simulation
│   ├── calibration.py      # fit to published marginals + check
│   ├── experiment.py       # A/B, ablation, ordering, substitution
│   ├── statistics.py       # summaries, bootstrap, subgroups, sensitivity
│   ├── visualization.py    # Plotly figures
│   ├── export.py           # CSV / JSON with provenance
│   └── disclaimer.py       # shared integrity text
├── tests/                  # 104 tests
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .streamlit/config.toml
└── .github/workflows/tests.yml
```

---

## 8. Tests

```bash
python -m pytest -q
```

Last run: **104 passed** on Python 3.12.

Coverage includes: citation integrity (every DOI and sample size, and a test that no
source claims a technique effect), parser behaviour including negation and the
"glans"-read-as-seconds regression, population heterogeneity and reproducibility,
model bounds and determinism, ordering effects (paced beats direct-first), removal
effects, calibration within tolerance, paired design, ablation completeness, export
contents and invalid input handling.

---

## 9. Limitations you must state if you present this

- No published study estimates the causal effect of a specific technique, pressure,
  rhythm or sequence. Every technique parameter is an assumption.
- Calibration fixes the level of the null scenario. It does not validate the
  mechanisms.
- The warm-up, overstimulation and rhythm mechanisms are plausible and consistent with
  the literature, but their functional forms are invented.
- The model cannot represent skill within a category: "gentle suction" is one number,
  not a technique.
- Ordering effects are the least supported part of the model. Treat any ranking of
  sequences as a hypothesis, not a finding.
- Bootstrap intervals describe simulation noise conditional on the model. They are not
  confidence intervals for anything real.
- One-at-a-time sensitivity cannot show what happens when several assumptions are
  wrong together.

**It cannot establish** that any guide works, any real-world effect size, any
prevalence or success rate, clinical validity, or anything about an individual person.

---

## 10. Changelog

**0.3.0** — sequence architecture; guide parser with review step; 1,000-woman
heterogeneous population; minute-by-minute model with warm-up, overstimulation, rhythm
and fit mechanisms; calibration to published marginals; ablation, ordering,
substitution and subgroup studies; evidence database with verified citations and
per-parameter evidence levels; 104 tests.

**0.2.0** — research-anchored ordinal satisfaction scenario, hypothetical effect
slider, research evidence section, export provenance.

**0.1.0** — synthetic couples, seven-variable protocol, hypothetical response model,
paired engine, statistics layer, dashboard, exports, 70 tests.

---

## Licence and use

Use it for teaching and for thinking clearly about what a model can and cannot say.
Do not present it as research findings, and do not describe it as validated or
clinically tested.
