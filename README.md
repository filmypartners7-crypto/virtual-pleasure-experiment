# Virtual Pleasure Experiment (VPE)

**Version 0.2.0** — a research-anchored computational scenario tool, packaged for
deployment on GitHub and Streamlit Community Cloud.

VPE generates 1,000 synthetic adult couples by default, lets an experimenter map a
guide to seven structured variables, and displays two clearly separated kinds of
output:

1. **Hypothetical response-model indices** for subjective arousal, pleasure,
   comfort, satisfaction and discomfort risk. Their coefficients and equations are
   assumptions, not fitted research models.
2. **An ordinal satisfaction scenario anchored to published human-survey data.**
   The starting category percentages are resampled from a published Spanish
   women's survey that asked about satisfaction with receiving oral sex
   (cunnilingus). The effect of the user's guide is an explicit hypothetical
   parameter, because that paper did not test the guide.

> **Interpretation rule.** The reference distribution is real published aggregate
> data. The 1,000 profiles are synthetic. The effect assigned to the guide is
> hypothetical. A modelled shift is not an observed improvement and not a
> real-world success rate.

---

## Contents

1. [What changed in 0.2.0](#1-what-changed-in-020)
2. [Research sources and how they are used](#2-research-sources-and-how-they-are-used)
3. [Project structure](#3-project-structure)
4. [How the empirical scenario works](#4-how-the-empirical-scenario-works)
5. [The continuous response model](#5-the-continuous-response-model)
6. [Statistics and uncertainty](#6-statistics-and-uncertainty)
7. [Requirements and supported versions](#7-requirements-and-supported-versions)
8. [Run locally](#8-run-locally)
9. [Deploy to Streamlit Community Cloud](#9-deploy-to-streamlit-community-cloud)
10. [Push to GitHub first](#10-push-to-github-first)
11. [Reproducibility and exports](#11-reproducibility-and-exports)
12. [Test suite](#12-test-suite)
13. [Troubleshooting deployment](#13-troubleshooting-deployment)
14. [Scientific limits and responsible wording](#14-scientific-limits-and-responsible-wording)
15. [Changelog](#15-changelog)

---

## 1. What changed in 0.2.0

- Added the 2022 Spanish survey's rounded aggregate satisfaction categories for
  receiving oral sex: 5.3% not practised, 3.1% not satisfactory, 10.5% somewhat
  satisfactory, 21.9% quite satisfactory, 59.2% very satisfactory.
- The aggregate categories are resampled to create a synthetic reference cohort.
  No participant-level records or personal data are included.
- Added a user-visible **hypothetical guide-effect coefficient**. Zero is the
  default and represents the no-effect scenario; positive and negative values
  explore assumed shifts. This coefficient is not estimated from any study.
- Added a research evidence page (dashboard section G), source descriptions,
  export provenance and tests for the empirical scenario.
- Kept the continuous pleasure/arousal/comfort equations separate and explicitly
  hypothetical.
- **Deployment hardening (this packaging pass):** bounded dependency ranges,
  `pytest` moved out of the runtime image, `.streamlit/config.toml` added,
  `pytest.ini` added, GitHub Actions CI added, secrets ignored, deployment
  documentation expanded. No scientific feature was removed and the mathematical
  model was not replaced or mocked.

---

## 2. Research sources and how they are used

### 2.1 Direct satisfaction reference (the only numeric import)

Ogallar-Blanco, Lara-Moreno & Godoy-Izquierdo (2022), *Going beyond "With a
Partner" and "Intercourse": Does Anything Else Influence Sexual Satisfaction among
Women? The Sexual Satisfaction Comprehensive Index*, **International Journal of
Environmental Research and Public Health**, 19(16), 10232.
DOI: [10.3390/ijerph191610232](https://doi.org/10.3390/ijerph191610232) ·
[Open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9408228/)

- Sample: 1,080 women aged 18–50 in Spain, recruited through a voluntary online
  non-probability sample.
- **Table 4** reports satisfaction categories for receiving oral sex
  (cunnilingus). VPE uses the rounded percentages as the starting ordinal
  reference distribution.
- **Table 3** reports factors respondents rated as affecting their sexual
  satisfaction "a lot". VPE displays these descriptively only. They are perceived
  influences, not causal effect sizes, and are not specific to cunnilingus.
- Limitations: self-selected sample, one national context, cross-sectional
  self-report. Not globally representative. It did not evaluate this guide.

**Transcription check.** The values embedded in `vpe/research.py` were verified
against the published tables on 17 November 2026:

| Field | Embedded | Published | Status |
| --- | --- | --- | --- |
| Table 4, receiving oral sex | 5.3 / 3.1 / 10.5 / 21.9 / 59.2 | identical | ✅ exact |
| Table 4 mean ± SD | — | 3.27 ± 1.11 [0–4] | ✅ consistent with those percentages (weighted mean 3.27) |
| Table 3, seven factor rows | 72.5 / 72.1 / 66.8 / 63.3 / 51.5 / 53.7 / 39.3 | identical | ✅ exact |

### 2.2 Preference variation

Herbenick et al. (2018), *Women's Experiences With Genital Touching, Sexual
Pleasure, and Orgasm: Results From a U.S. Probability Sample of Women Ages 18 to
94*, **Journal of Sex & Marital Therapy**, 44(2), 201–212.
DOI: [10.1080/0092623X.2017.1346530](https://doi.org/10.1080/0092623X.2017.1346530)

A U.S. probability sample of 1,055 women documented variation in genital-touch
preferences. The often-quoted clitoral-stimulation percentages (36.6% needed
clitoral stimulation for orgasm during intercourse; 36.0% said it made orgasms
feel better; 18.4% found penetration alone sufficient) concern orgasm during
intercourse, not a specific cunnilingus technique. VPE uses this study only to
support representing preference heterogeneity. It imports no technique-effect
coefficient.

### 2.3 Context and negotiation

Backstrom, Armstrong & Puentes (2012), *Women's Negotiation of Cunnilingus in
College Hookups and Relationships*, **Journal of Sex Research**, 49(1), 1–12.
DOI: [10.1080/00224499.2011.585523](https://doi.org/10.1080/00224499.2011.585523)

A qualitative interview study of 43 college women. It supports the relevance of
relationship context and negotiation, but it is not a source for population
percentages or numeric effect sizes.

### 2.4 Separate dimensions of sexual function

Rosen et al. (2000), *The Female Sexual Function Index (FSFI): A Multidimensional
Self-Report Instrument for the Assessment of Female Sexual Function*, **Journal of
Sex & Marital Therapy**, 26(2), 191–208.
DOI: [10.1080/009262300278597](https://doi.org/10.1080/009262300278597)

The FSFI measures six domains: desire, subjective arousal, lubrication, orgasm,
satisfaction and pain. It supports keeping dimensions separate; it provides no
coefficients for cunnilingus or for this guide.

> Only source 2.1 supplies numbers to the model. Sources 2.2–2.4 justify design
> choices and are displayed as descriptive context in dashboard section G.

---

## 3. Project structure

```text
vpe/                              <- repository root (contains app.py)
├── app.py                        # Streamlit dashboard, sections A-G
├── README.md
├── requirements.txt              # runtime dependencies (Streamlit Cloud reads this)
├── requirements-dev.txt          # runtime + pytest
├── pytest.ini                    # test discovery config
├── .gitignore                    # ignores caches, venvs and secrets
├── .streamlit/
│   └── config.toml               # theme, no telemetry, headless server
├── .github/
│   └── workflows/
│       └── tests.yml             # CI: pytest on Python 3.11 and 3.12
├── vpe/
│   ├── __init__.py
│   ├── model_config.py           # continuous-model coefficients and evidence labels
│   ├── disclaimer.py             # shared caveats used by app and exports
│   ├── population.py             # synthetic adult profiles + empirical category anchor
│   ├── protocol.py               # seven structured guide variables and baseline convention
│   ├── research.py               # study provenance and published aggregate data
│   ├── empirical_scenario.py     # ordinal reference distribution + hypothetical shift
│   ├── response_model.py         # hypothetical continuous response equations
│   ├── experiment.py             # paired model and empirical scenario orchestration
│   ├── statistics.py             # summaries, bootstrap intervals and sensitivity
│   ├── visualization.py          # Plotly charts
│   └── export.py                 # CSV / JSON exports with provenance
└── tests/
    ├── conftest.py
    ├── test_population.py
    ├── test_response_model.py
    ├── test_experiment.py
    ├── test_statistics.py
    ├── test_research.py
    └── test_empirical_scenario.py
```

---

## 4. How the empirical scenario works

The source paper provides aggregate satisfaction category percentages, not raw
individual responses, and not an experimental effect for this guide.

1. VPE samples 1,000 synthetic category labels using the source's rounded
   percentages.
2. For participants classified as having practised receiving oral sex, it samples a
   latent logistic utility within the corresponding published ordinal category.
   This latent value is a mathematical device, not a measured survey variable.
3. VPE calculates the guide's structured score as the equal-weight mean of the
   seven 0–1 sliders.
4. It applies the explicit hypothetical coefficient:

   ```text
   latent_shift = assumed_effect_logit * (guide_mean_slider_score - baseline_mean_slider_score)
   ```

5. The shifted latent value is mapped back to the same ordinal category
   cut-points.

The coefficient is adjustable from −3 to +3. **Zero is the default and means no
assumed guide effect.** Positive and negative values are sensitivity scenarios.
Because no cited study estimates the effect of this particular guide, the
application must not describe these shifts as research-derived predictions.

"Not practiced" profiles remain in that category, so an assumed technique effect
cannot silently change whether the activity was practised.

---

## 5. The continuous response model

The app separately generates hypothetical indices for subjective arousal,
pleasure, comfort, satisfaction and discomfort risk, using logistic
latent-variable equations with synthetic profile inputs and configurable
coefficients. These coefficients are not calibrated to participant-level data.
Sensitivity analysis therefore describes how the equations respond to assumptions,
not how actual people would respond.

Subjective arousal is one moderate input to pleasure, never its definition.
Discomfort risk is a separate mechanism driven by boundary lapses, absent comfort
checks, communication gaps and instability — not `1 − comfort`. Any orgasm-related
estimate is opt-in, off by default, and labelled as an unvalidated hypothetical
output.

The guide text field is for reference only. **VPE does not parse the pasted
guide.** The experimenter manually maps the guide to the seven structured
variables, which avoids pretending a free-text description has been
scientifically interpreted.

---

## 6. Statistics and uncertainty

The dashboard reports means, medians, standard deviations, quantiles,
distributions, within-couple paired differences, bootstrap percentile intervals
conditional on the model, subgroup comparisons and one-at-a-time sensitivity
analysis.

Four kinds of uncertainty are kept separate and are never blended:

| Kind | Status |
| --- | --- |
| Model output | deterministic given inputs, coefficients and noise |
| Statistical uncertainty (conditional on the model) | bootstrap intervals over synthetic couples only |
| Uncertainty from assumptions | larger than the intervals and **not quantified** |
| Real-world empirical evidence | **absent**; running the simulation does not create it |

Intervals are labelled "bootstrap percentile interval (conditional on the model)"
everywhere, in the app and in exports. They are never called confidence intervals.

---

## 7. Requirements and supported versions

**Python 3.11 or newer.** Tested on Python 3.12.13. Every module uses
`from __future__ import annotations`, so the type hints do not require a specific
interpreter version, but `pandas` 3.x requires 3.11+.

Runtime (`requirements.txt`):

| Package | Range | Tested | Role |
| --- | --- | --- | --- |
| streamlit | >=1.32,<2 | 1.67.0 | dashboard |
| numpy | >=1.26,<3 | 2.5.4 | sampling and linear algebra |
| pandas | >=2.0,<4 | 3.1.1 | tabular data |
| plotly | >=5.18,<8 | 7.1.0 | interactive charts |
| scipy | >=1.11,<2 | 1.18.1 | part of the declared stack; not imported by current runtime code |

Development (`requirements-dev.txt`): the above plus `pytest>=7.4,<10` (tested on
9.2.0).

Ranges are bounded on both ends so a future major release cannot silently change
behaviour in a deployed app. A fresh resolution of `requirements.txt` was verified
to complete without conflicts (36 packages, no resolver errors).

No database, authentication, external API, paid service or secret is required.

---

## 8. Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>). The default
population is 1,000 couples; set a seed to reproduce a run exactly.

---

## 9. Deploy to Streamlit Community Cloud

You need a GitHub account and a Streamlit Cloud account (sign in with GitHub).

### 9.1 The one mistake that breaks this deployment

Streamlit Cloud runs `app.py` **from the repository root** and reads
`requirements.txt` **from the repository root**. Your repository root must be the
folder that *contains* `app.py`, not a parent folder that contains a `vpe/`
directory holding it.

Correct:

```text
your-repo/
├── app.py            ✅ at the root
├── requirements.txt  ✅ at the root
└── vpe/              ✅ the package
```

Wrong (app will fail with "main file does not exist" or missing modules):

```text
your-repo/
└── vpe/              ❌ app.py is nested one level too deep
    ├── app.py
    └── requirements.txt
```

So: `cd` into the folder that contains `app.py` **before** you run `git init`.

### 9.2 Deploy

1. Push the project to GitHub (section 10).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository**: `<your-user>/<your-repo>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Open **Advanced settings**:
   - **Python version**: choose **3.12** (or 3.11). Do not pick 3.9 — `pandas` 3.x
     needs 3.11+, and the app is only tested on 3.11/3.12.
   - **Secrets**: none required. Leave empty.
6. Click **Deploy**. The first build installs `requirements.txt` and typically
   takes 2–5 minutes.
7. When the app loads, press **Run experiment** to confirm the engine executes in
   the cloud. You should see 1,000 synthetic couples, 13 charts and two download
   buttons.

### 9.3 Post-deploy checks

- [ ] The page loads with the red "Research anchor, synthetic participants" banner.
- [ ] Sections A–G all open.
- [ ] **Run experiment** returns a run fingerprint and populates sections B–F.
- [ ] The sensitivity button completes without an error.
- [ ] Both downloads work and the CSV's first lines carry the `# SYNTHETIC DATA`
      disclaimer.
- [ ] With the guide-effect slider at 0.0, the empirical scenario reports 100%
      unchanged — the null scenario must stay null.

### 9.4 Updating a deployed app

Push to the same branch; Streamlit Cloud rebuilds automatically. Apps sleep after
inactivity and wake on the next visit. To pin a release, deploy a tag or a
separate branch instead of `main`.

---

## 10. Push to GitHub first

From inside the folder that contains `app.py`:

```bash
git init
git add .
git commit -m "VPE 0.2.0 - research-anchored scenario tool"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Check before pushing:

```bash
git status --short          # nothing sensitive staged
git ls-files | head -30     # app.py and requirements.txt must be at the root
```

`.gitignore` already excludes `__pycache__/`, `.pytest_cache/`, virtual
environments and `.streamlit/secrets.toml`.

**CI.** `.github/workflows/tests.yml` runs the full test suite on Python 3.11 and
3.12 for every push to `main` and every pull request. A green check is your
evidence that the deployed code passes its tests — do not claim tests pass
without looking at the run.

---

## 11. Reproducibility and exports

- Same seed + same protocol + same coefficients + same guide-effect setting ⇒
  identical results.
- A 16-character **run fingerprint** hashes the full settings so you can tell
  whether two runs share a configuration.
- **CSV export**: one row per couple, with profile variables, both conditions'
  outcomes, paired differences and the empirical reference category. The
  synthetic-data disclaimer is the file header (comment lines starting with `#`).
- **JSON export**: model version, seed, population size, protocol, baseline
  definition, coefficients with evidence labels, the equations, research
  provenance (including the source study metadata and the data-handling
  statement), the empirical scenario metadata and the disclaimer.

---

## 12. Test suite

```bash
python -m pytest -q
```

Last verified run: **88 passed** on Python 3.12.13 with the pinned ranges above.

Coverage includes:

| Area | Checks |
| --- | --- |
| Population | exactly 1,000 couples by default, adults only, bounded profiles, seed reproducibility, seed sensitivity, no personal identifiers, invalid sizes rejected |
| Response model | outcomes inside (0,1) even with extreme coefficients, determinism, better protocol ⇒ better outcomes, arousal ≠ pleasure, discomfort ≠ 1 − comfort, opt-in orgasm output, invalid protocol/coefficient rejection |
| Empirical scenario | published percentages sum to 100%, resampling reproduces the marginals, null effect at 0.0, positive/negative shifts move categories the right way, "not practiced" held fixed, invalid effect rejected |
| Research | citation metadata complete, tables render, plus **citation-lock** tests that pin the published Table 3 and Table 4 values so any transcription drift fails CI |
| Experiment | identical paired profiles, reproducible runs, different seeds differ, settings record, exports contain configuration and disclaimer |
| Statistics | hand-computed means/medians/SDs/quantiles, paired differences, bootstrap reproducibility and interval ordering, sensitivity coverage, subgroup partitioning, invalid arguments |

---

## 13. Troubleshooting deployment

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Error: main file does not exist` | main file path wrong, or `app.py` nested inside a subfolder | Set the path to `app.py` and make the repo root the folder containing it |
| `ModuleNotFoundError: vpe` | the `vpe/` package was not pushed (ignored by a broad pattern) | `git ls-files vpe/` and commit the package |
| `ModuleNotFoundError: streamlit` | `requirements.txt` missing or not at the repo root | Confirm `requirements.txt` sits beside `app.py` |
| Build fails resolving packages | a local `pip freeze` pinned versions unavailable for the cloud Python | Use the provided ranges, not exact pins from your laptop |
| App loads but charts are blank | stale browser cache after a rebuild | Hard-refresh; the chart code is exercised by the tests |
| App sleeps / slow first load | free tier sleeps after inactivity | Expected; the first visit wakes it |
| `KeyError` on a session-state value | you clicked a section before running an experiment | Press **Run experiment** in section A first |

---

## 14. Scientific limits and responsible wording

VPE can show the published aggregate reference categories, explore hypothetical
scenarios, and reveal which assumptions drive a model's output. It **cannot**:

- prove that the guide works, or estimate how well it works;
- identify a universal best technique;
- predict any individual woman's response;
- establish real-world causal effects;
- support a claim that a cited study tested this guide — none did.

Good wording for a video:

> "I built a research-anchored virtual experiment. Its reference distribution
> comes from a published survey, while the effect of my guide is explored as a
> hypothetical scenario. The results help inspect assumptions and evidence gaps;
> they are not real-world success rates."

Avoid wording such as "the study proves my guide works for X% of women." No
included study tested this guide, and the effect coefficient is a user-chosen
assumption, not a finding.

---

## 15. Changelog

**0.2.0** — research-anchored ordinal satisfaction scenario; published aggregate
reference distribution; user-visible hypothetical guide-effect coefficient;
research evidence section G; export provenance; new tests for research and the
empirical scenario; deployment hardening (bounded ranges, dev requirements split,
Streamlit config, pytest config, CI workflow, expanded README).

**0.1.0** — synthetic population generator, seven-variable protocol builder,
hypothetical continuous response model, paired experiment engine, statistics
layer, six-section dashboard, CSV/JSON export, 70 tests.

---

## Licence and use

Use it for teaching, for demonstrating simulation workflow, and for thinking
clearly about what a model can and cannot say. Do not present it as research
findings, and do not describe it as validated or clinically tested.
