Overview
Each example is one human protein, given to you in two complementary modalities:

An immunofluorescence microscopy image, a 2-channel image of cultured human cells in which one target protein has been antibody-stained (channel 0, "protein") alongside a nucleus reference stain (channel 1, "nucleus"). The image shows where the protein appears inside the cell. It deliberately ships only these two channels, the microtubule and endoplasmic-reticulum reference channels that normally anchor the spatial frame are not provided.
A functional text description, a short prose summary of what the protein does (its molecular function, the biological processes it participates in, its protein family/domains). Every statement about where the protein is located has been removed: no compartment names, no subcellular-location sentences, no localization terms. What remains is a functional prior over plausible compartments, never the answer.
Your job is to predict the protein's set of subcellular locations, a multi-label prediction over a fixed vocabulary of 16 compartments, from the image and the text together.

This is a multi-label classification problem: a protein may localize to one or several compartments at once (many proteins have a dominant location plus one or more secondary locations). You output an independent probability in [0, 1] for each of the 16 classes.

The 16 location classes (these are the exact column suffixes, see Submission format):

nucleoplasm  nuclear_membrane  nucleoli  nucleoli_fibrillar_center  nuclear_speckles
nuclear_bodies  endoplasmic_reticulum  golgi_apparatus  vesicles  plasma_membrane
cytosol  mitochondria  microtubules  centrosome  actin_filaments  intermediate_filaments

The intended solution is a deep multimodal model, and there are two approved architectures: (a) a vision encoder plus a text encoder, fused into a 16-way multi-label (sigmoid) head and trained end to end on the training pairs; or (b) a vision-language model (VLM) fine-tuned to read the 2-channel image and the function text jointly and emit the 16 labels. Classical ML is not an approved approach, predictions must come from a learned deep model over the raw image and raw text, not from gradient-boosted trees / random forests / SVMs / logistic regression on extracted or hand-crafted features.

Why both modalities are required
This is a fusion task by construction. The full set of a protein's locations depends on both what its staining pattern looks like (visible only in the image) and what the protein does (stated only in the text), and the two inputs were split so that neither is sufficient alone:

The image shows the dominant staining pattern, but with only the protein + nucleus channels (no microtubule/ER reference frame) many compartments become visually ambiguous, ER vs diffuse cytosol, Golgi vs vesicles vs peri-nuclear signal, the fine sub-nuclear patterns, and secondary locations are often faint and easy to miss. From the image alone you will systematically under-call the secondary compartments.
The text tells you the protein's function and family, a strong prior on which compartments are plausible (a spliceosome component leans nuclear; a glycosyltransferase leans secretory; a motor protein leans cytoskeletal), but it never states the actual pattern or the specific compartments, and all location words are stripped.
A model that reads only one modality will leave a large amount of signal on the table, especially on the secondary locations, which is where the metric is won. Strong solutions genuinely combine the visual pattern with the functional prior.

This is a generalisation challenge
A naive classifier will not do well; the difficulty is deliberate.

Unseen protein families at test. The test set is split by protein family: every test protein's family is absent from training. You cannot memorise "this family localizes here" from the text, you must learn how the functional prior combines with the visual pattern in a way that transfers to novel families. The train-only family_group_id lets you reproduce this split locally (leave-family-out CV).

Enriched for the hard cases. The test set is enriched for multi-localizing proteins (two or more compartments) and visually-confusable patterns, exactly the regime where the image alone is insufficient and the functional text earns its keep.

Intrinsically variable labels. Subcellular annotations carry genuine biological and annotation variability (reliability tiers, partial/secondary calls), so there is an irreducible error floor, a perfect score is not attainable, and that is expected.

Realistically, expect a fused model to reach raw macro-F1 ≈ 0.45 to 0.58 (and a score, see Evaluation, of about 0.40 to 0.50). A constant/most-common-set prediction scores ≈ 0.

Dataset
All files live in the working directory. Images are NumPy arrays under train/ and test/.

The image (.npy)
Each train/<id>.npy / test/<id>.npy is a uint8 array of shape (2, 320, 320), (channel, height, width). Channel order is fixed: channel 0 = target protein, channel 1 = nucleus. Load with numpy.load. The two reference channels (microtubules, ER) are not included by design. Pixel intensities carry a small incidental jitter/noise from the build; treat it as augmentable.

train.csv
One row per training protein-image. Columns, in order:

id, string. Opaque 16-char hex id; matches train/<id>.npy.
npy_path, string. Path to the image relative to the dataset root, i.e. train/<id>.npy.
function_text, string. The location-stripped functional description (function summary + molecular-function / biological-process annotations + family). Location words and all identifiers are masked ([LOC], [ID]).
family_group_id, integer. An opaque group id shared by training proteins of the same family. It carries no biological meaning and exists only so you can run leave-family-out cross-validation that mimics the train→test shift. It is not present in test.csv.
loc_<class>, 16 columns, one per location class (loc_nucleoplasm, …, loc_intermediate_filaments). The gold multi-hot label, each in [0, 1] (training labels carry a small privacy noise; treat them as soft targets).
test.csv
One row per test protein-image. Columns, in order: id, npy_path, function_text. There is no loc_* column (that is what you predict) and no family_group_id (test families are all unseen). id is an opaque hash; the gene, antibody, and database identifiers are not recoverable from it or from any shipped column.

sample_submission.csv
A correctly-formatted starter file. Each loc_<class> column is pre-filled with that class's training prevalence (a constant). Replace the values with your predicted probabilities and save as submission.csv.

location_labels.json
The index → class-name map (the frozen column order), for convenience.

Evaluation
The score is a prior-anchored macro-F1 skill score in [0, 1]. First, per-class F1 at a threshold of 0.5 is computed and macro-averaged over the location classes that appear in the gold (each class counts equally, so under-calling rare/secondary classes is penalised):

RawMacroF1 = mean over classes c (with >=1 gold positive) of  F1_c( pred>=0.5 , gold )

Then it is anchored against the constant most-frequent-set baseline F1_base (predicting, for every protein, exactly the locations whose training prevalence ≥ 0.5):

LocSkill = clip( (RawMacroF1 - F1_base) / (1 - F1_base),  0, 1 )

Direction: maximize, range [0, 1].

Predicting the most-common location(s) for everything → RawMacroF1 ≈ F1_base → LocSkill ≈ 0.
Every protein's location set exactly right → LocSkill = 1.
Predictions are matched to gold by id. A missing row, or NaN/missing values in a loc_<class> cell, are treated as absent (0) for that cell, not a crash.
The returned score is floored at 0.01 (never exactly 0).
Duplicate ids, a missing id column, or a submission with none of the loc_* columns cause the submission to be rejected. Extra ids not in the test set are ignored.
Submission format
Produce submission.csv in your output directory with the id column plus the 16 loc_<class> probability columns, one row per test.csv protein. Header (exact column names, any column order is fine as long as names match):

id,loc_nucleoplasm,loc_nuclear_membrane,loc_nucleoli,loc_nucleoli_fibrillar_center,loc_nuclear_speckles,loc_nuclear_bodies,loc_endoplasmic_reticulum,loc_golgi_apparatus,loc_vesicles,loc_plasma_membrane,loc_cytosol,loc_mitochondria,loc_microtubules,loc_centrosome,loc_actin_filaments,loc_intermediate_filaments
a1b2c3d4e5f60718,0.91,0.02,0.10,0.01,0.04,0.03,0.05,0.02,0.06,0.01,0.44,0.02,0.01,0.01,0.01,0.01

Each loc_<class> is a probability in [0, 1]; the grader thresholds at 0.5.
Match is by id alone; row order does not matter.
Any test id you omit is scored as all-absent. Do not submit duplicate ids.
Hints and constraints
Intended approach (two approved options). Either (a) a two-encoder fusion model, an image encoder (e.g. an ImageNet-pretrained ConvNeXt-Tiny / EfficientNetV2-S with its first conv adapted to 2 input channels) over the .npy image and a learned text encoder (e.g. a biomedical BERT such as PubMedBERT/BioBERT, or DistilBERT) over function_text, fused (concatenation or cross-attention) into a 16-way sigmoid head; or (b) a vision-language model (VLM) fine-tuned (e.g. with QLoRA) to read the image + text jointly and emit the 16 labels. Every loc_<class> value must come from your trained deep model.
Classical ML is not approved. Gradient-boosted trees (LightGBM/XGBoost), random forests, SVMs, k-NN, or logistic regression, whether on hand-crafted features or stacked on frozen image/text embeddings, are not an accepted predictor here. The 16 outputs must be produced by a neural multimodal model (two-encoder fusion or a VLM) that learns from the raw image and raw text.
Use both modalities. A single-modality model is a weak baseline here by design. Verify fusion helps by comparing against image-only and text-only ablations. The secondary-location recall, where the functional prior pays off, is what moves macro-F1.
Multi-label, not single-label. A protein may have several locations: use independent per-class outputs (per-class sigmoids, or a VLM that emits the full label set), not a single-choice softmax. Handle class imbalance (class-balanced/focal loss; per-class thresholds) and protect the rare classes, macro-F1 weights them equally with the common ones.
Generalise across families. Test families are unseen. Use family_group_id to build a leave-family-out validation split so your internal estimate reflects real transfer.
The text is function, not location. Don't try to recover the compartment from the prose, it is stripped. Read the enzyme class, partners, pathway, and family; that is the prior.
Don't hardcode. Lookup tables, regex, if/elif chains, or templates that set any loc_<class> from id, from string matches in function_text, or from metadata are not valid, every predicted value must come from your trained model's output.