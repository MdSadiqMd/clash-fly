# Dognosis GC-MS data: what it contains and what it can support

Written 2026-09-12 from data/raw_data/Dognosis_Batch_Compilation_313.xlsx
(Sheet1) and data/all_cancer_data/*.xlsx. The 224 workbook is a strict subset
of 313 (56 of its file codes, 13,134 of its columns), so only 313 is used.

## Structure

- 88 breath-mask samples (file codes), 7 labels: Chronic 33, Healthy 13,
  Oral Cancer 13, Oesophageal Cancer 11, Benign 7, Breast Cancer 6, Throat 5.
  Throat is treated as cancer because 13+11+6+5 = 35 matches the
  "Detection_Incubation = 35" count in the all-cancer sheets. Chronic and
  Benign are treated as non-cancer. Cancer 35, non-cancer 53.
- 4 rows per sample: {Incubation, No Incubation} x {plain, IS}. IS rows carry
  only the internal standard (p-bromofluorobenzene, RT 10.47) area; the plain
  rows carry the data. Some samples lack no-incubation runs (80 vs 88).
- 14,075 columns = NIST library hit name + retention time. Raw peak areas,
  259 to 8.8e8. Density 0.5%: a run has median 27 detected compounds
  (range 0 to 270). Collapsing retention-time variants by name gives 9,776
  names; 274 are detected in >=10% of incubation runs, 47 in >=20%.
- Dominant compounds: n-alkanes and branched alkanes (dodecane,
  tetradecane, pentadecane, 4,6-dimethyldodecane), siloxanes (column bleed
  or mask/bag), 2,4-di-tert-butylphenol (plastic antioxidant), adamantyl
  esters and halogenated names that are library mis-hits. Nonanal is the only
  common compound with a plausible biological origin and fly receptor data.
- all_cancer_data sheets compare Incubation vs No Incubation inside the cancer
  group. They describe the incubation effect (alkanes rise), not cancer vs
  control. Not usable as a cancer signature.

## Result 1: a plain classifier separates cancer from control almost perfectly

Leave-one-sample-out, logistic regression on presence/absence of the 274
common compounds, incubation runs: AUC 0.987. Shuffled labels give 0.35-0.56.
No-incubation runs: 0.985. Random forest and log-area features agree.

## Result 2: that separation is a batch signature, not chemistry of disease

| Feature set (incubation runs) | LOO AUC |
|---|---|
| all 274 | 0.987 |
| artefacts only (63 siloxanes, silanes, TMS derivatives, adamantyl and halogenated mis-hits) | 0.994 |
| non-artefact only | 0.977 |
| non-artefact minus alkanes | 0.916 |
| DoOR-named compounds only (16) | 0.896 |

Unsupervised clustering (Jaccard, average linkage) on artefact features only,
no labels used, splits the samples into a 32-sample cluster containing 32 of
the 35 cancers and zero controls, and a 51-sample cluster containing 50
controls plus 3 oesophageal cancers. The same split appears with
non-artefact features and in the no-incubation runs.

Supporting evidence:
- Healthy vs Chronic controls separate with AUC 0.95-1.00. Two control
  groups that should be chemically similar are not, so site or handling
  differs between control sources too.
- Detected-compound count per run: Oral, Throat, Healthy, Benign ~200-230;
  Breast, Oesophageal, Chronic ~90-110. Handling or run-time differences.
- Internal standard area differs by class (cancer median 2.0e7 vs 1.5e7,
  Mann-Whitney p = 0.02) and by group in both conditions. The instrument
  itself saw the groups differently.
- Specific markers: cyclotetrasiloxane detected in 97% of cancer vs 41% of
  controls; tetradecane in 0% of cancer no-incubation runs vs 78% of
  controls; 1,2-dichlorobenzene 0% of oesophageal and breast vs 77-100% of
  Healthy and Benign.

Conclusion: cancer and non-cancer samples were collected, stored, or run in
different batches, and the label is almost perfectly confounded with batch.
Any model, fly circuit or XGBoost, trained on this table will learn the batch
and report a high AUC that means nothing about cancer.

## Result 3: the fly cannot see this chemistry anyway

Of the 274 common compounds, 16 have a DoOR entry. Of those, only nonanal
(29 receptors measured, max response 0.37), decanal (20), 3-carene (28),
2-propenal (24) and toluene (6) have real receptor data. The alkanes that
dominate the table have 0 or 1 receptor measured with responses near 0.02:
fly olfactory receptors do not respond to C10-C20 alkanes. Compounds the fly
does respond to strongly (benzaldehyde, hexanal, 2-butanone, acetone) are
detected in under 6% of runs here.

So the GC-MS-to-ORN step for this dataset would be almost entirely QSAR
extrapolation onto chemistry outside DoOR's training space, and the answer
it would give is "the fly smells nothing". That is a finding, not a bug.

## What would make the fly project real with Dognosis data

1. Batch metadata per sample: collection site, hospital, mask lot, bag lot,
   collection date, GC-MS run date and sequence position. Then either
   stratify or show the cancer signal survives within batch. Without this
   the dataset supports no cancer claim.
2. Controls collected at the same sites with the same masks on the same days
   as cancers. Even 10 such controls would let the confound be tested.
3. Blank masks (unworn, same lot) run through the same pipeline to subtract
   siloxanes, phenol antioxidant and alkane background.
4. Peak table with CAS or InChIKey and match scores, not NIST names, so the
   collapsed compound identity is trustworthy and can be joined to DoOR.
5. Lower-mass volatiles retained: the current method loses the aldehydes,
   ketones and esters a fly nose is built for. SPME fibre choice or a
   different desorption profile matters more than any model.

## What can still be done now, honestly

- Publish the confound analysis itself (this file). It is the correct
  first result and protects the later claim.
- Run the fly pipeline on the Louisville carbonyl panel (data/raw/breath),
  which has 427 subjects and compounds the fly responds to, as the
  methods demonstration. Present the Dognosis data as "pending batch
  metadata".
- Optionally run the fly pipeline on Dognosis nonanal + decanal + 3-carene +
  toluene only, with the explicit statement that this is the fly-visible
  subset and that it carries batch signal (AUC 0.90 on DoOR-named compounds
  above).
