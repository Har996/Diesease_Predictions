# Dataset folder

`disease_symptoms.csv` is the file `python -m src.train` reads.

## The file shipped with this project is SYNTHETIC

It was produced by `python -m src.generate_sample_data` from hand-written symptom probabilities.
It exists only to demonstrate the pipeline. `disease_symptoms.meta.json` marks it as synthetic
(it stores the file's SHA-256, so the app stops calling the data "synthetic" as soon as you replace the CSV).

## Using a real dataset

Replace `disease_symptoms.csv` (and delete `disease_symptoms.meta.json`), then run `python -m src.train`.
Two layouts are detected automatically; column names are case- and punctuation-insensitive.

**Wide** - one 0/1 column per symptom (yes/no/true/false also work):

| fever | cough | ... | age (optional) | gender (optional) | disease |
|------:|------:|-----|---------------:|-------------------|---------|
| 1 | 0 | ... | 34 | female | Influenza |

The target column may be called `disease`, `prognosis`, `diagnosis`, `condition` or `label`.

**Long** - symptom *names* in numbered columns (e.g. the Kaggle "Disease Symptom Prediction" file):

| Disease | Symptom_1 | Symptom_2 | Symptom_3 |
|---------|-----------|-----------|-----------|
| Fungal infection | itching | skin_rash | nodal_skin_eruptions |

Optional `age` (0-120) and `gender`/`sex` columns are used automatically if present.

## `disease_info.json`

General, non-personalised educational text shown on the result page (summary, commonly associated
features, when to seek care). Keys are disease names; diseases without an entry get generic text.
It deliberately contains no medication or treatment advice. Have any text you add reviewed by a clinician.
