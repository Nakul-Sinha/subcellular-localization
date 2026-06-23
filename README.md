# Protein Subcellular Localization

## The problem

For each human protein I get two very partial views and have to name every
compartment it lives in, as a multi-label choice over 16 compartments.

The first view is a two-channel immunofluorescence image: the stained target
protein, plus a nucleus reference. The microtubule and endoplasmic reticulum
reference channels that would normally anchor the spatial frame are deliberately
withheld. The second view is a short prose description of what the protein does,
with every statement about where it sits stripped out. So the text is a prior
over plausible compartments and never the answer.

Scoring is a prior-anchored macro F1 skill score under leave-family-out
splitting, so I am judged on protein families I have never seen.

## What I did

The task is a fusion problem, and both modalities are incomplete on purpose, so
neither one carries it alone. The image says where the signal is but without the
usual spatial reference frame, and the text says what the protein does but is
forbidden from saying where. The work is in combining them and in handling a long
tailed label distribution where rare compartments are exactly the ones the macro
F1 cares about.

## Layout

`solution.py` and `solution_core.py` are the entry points, `approach.md` is the
engineering map, and `requirements.md` lists the models and compute. Datasets are
not committed.
