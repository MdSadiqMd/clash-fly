# Cancer-sniffing fly: feasibility and build plan

Status: research only, nothing built. Written 2026-09-12.

## Verdict

Buildable. Every input exists, is public, and is CC-BY or MIT. The weak link is
mapping a breath VOC panel onto olfactory receptor neuron (ORN) activity; it is
weaker than the original note assumed in one way (breath data gives molecular
formulas, not compound names) and stronger in another (most of the identified
compounds are already measured in DoOR, so far less QSAR than feared).

## Data inventory (all downloaded to data/raw/, gitignored)

| Piece | Source | Format | Verified |
|---|---|---|---|
| Female brain edge list, v783 | Zenodo 10676866 (FlyWire, CC-BY); same release also ships as Connectivity_783.parquet (96 MB, >=5 synapses, 15.1M edges) inside the Shiu repo | feather / parquet | olfactory subgraph extracted, see below |
| Neuron labels | github flyconnectome/flywire_annotations | TSV, 139,248 neurons | 53 ORN glomerulus types, 685 ALPN, 5,177 KC, 96 MBON (35 types), 331 DAN (27 types), 2 APL |
| Odorant responses | DoOR 2.0, github ropensci/DoOR.data | door_response_matrix.csv 691 x 78, InChIKey rows; odor.csv has SMILES; door_mappings.csv maps receptor to glomerulus | yes |
| Breath panel | github sam-uofl/Dr.FuLab (Rai 2022 PLOS One) | LungCancer.txt, 427 subjects x 27 carbonyl formulas, nmol/L; Control 193 / Cancer 157 / Benign 77 | yes |
| Whole-brain LIF simulator | github philshiu/Drosophila_brain_model (Shiu 2024 Nature, MIT) | Brian2, activates neurons by FlyWire ID | cloned |
| Male CNS annotations | male-cns.janelia.org v1.0 (CC-BY) | feather, soma positions for demo point cloud | downloading |

User-supplied Dognosis GC-MS breath-mask data (data/raw_data, 88 samples,
35 cancer) was analysed on 2026-09-12: cancer and control samples sit in
separate collection or processing batches, and fly-visible compounds are
nearly absent. See docs/data-findings.md. Until batch metadata or same-site
controls exist, the fly pipeline should be demonstrated on the Louisville
panel and the Dognosis data reported as confounded.

Not GC-MS. The Louisville panel is carbonyl-selective capture on a silicon
microreactor read by FT-ICR-MS. It is the only open, labelled, per-subject
cancer breath dataset found. Metabolomics Workbench has no cancer breath study;
MetaboLights search failed. Say "breath carbonyl panel" in the writeup.

## Formula to compound to receptor

The panel columns are formulas. Identities from Fu 2014 (Cancer Med):

| Formula | Identity | In DoOR | Receptors measured |
|---|---|---|---|
| CH2O | formaldehyde | no | 0 |
| C2H4O | acetaldehyde | yes | 33 |
| C3H6O | acetone | yes | 48 |
| C4H8O | 2-butanone (cancer marker) | yes | 49 |
| C5H10O..C12H24O | n-alkanals pentanal..dodecanal (Fu-lab convention; isomer ambiguous) | yes | 41,41,21,23,29,20,1,1 |
| C13H26O | tridecanal | no | 0 |
| C4H8O2 | 3-hydroxy-2-butanone (cancer marker) | yes | 12 |
| C2H4O2 | 2-hydroxyacetaldehyde (cancer marker) | no; DoOR only has acetic acid | 0 |
| C3H4O | acrolein | yes | 24 |
| C6H10O2 | 4-hydroxyhexenal (cancer marker) | no | 0 |
| C9H16O2 | 4-hydroxynonenal | no | 0 |
| C7H6O | benzaldehyde | yes | 64 |
| C4H6O2 | 2,3-butanedione (assumed) | yes | 62 |
| others (C3H4O2, C4H6O, C4H4O2, C5H8O, C7H11O, C13H22O, C15H10O) | unresolved | mostly no | 0-2 |

So roughly 14 of 27 columns have measured receptor responses; 3 of the 4
published cancer markers do not (2-hydroxyacetaldehyde, 4-HHE, and partially
3-hydroxy-2-butanone with 12 receptors). Those need QSAR prediction from
DoOR-trained models (RDKit descriptors, per-receptor ridge or random forest,
leave-one-odorant-out error reported). This is the fiction the note warned
about, and it lands exactly on the diagnostic compounds. State it up front.

## Circuit extraction (FlyWire v783)

1. Filter edge list to pre/post in {ORN, ALLN, ALPN uniglomerular, APL, KC, MBON, DAN}.
2. ORN type name is the glomerulus (ORN_DM1 etc.). Join to DoOR via
   door_mappings.glomerulus, giving a 53-glomerulus x odorant response table.
3. Keep uniglomerular PNs only for the first version. Multiglomerular PNs go
   mostly to lateral horn, not KCs.
4. Build sparse matrices: W_orn_pn, W_pn_kc (expect mean ~6 PN claws per KC),
   W_kc_apl and W_apl_kc, W_kc_mbon, and the DAN to MBON-compartment map.
5. Sanity assertions: KC count ~5,000 in FlyWire (not 2,000; that is per
   hemisphere in older estimates), APL contacts nearly all KCs, 15 MBON
   compartments recovered from DAN innervation.

Measured on the v783 edge list (>=5 synapses per connection), both hemispheres:

| Block | Edges | Note |
|---|---|---|
| ORN -> ALPN | 19,438 | 53 ORN glomerulus types; 69 uPN glomerulus labels (includes thermo/hygro) |
| uPN -> KC | 26,075 | 4,820 of 5,177 KCs receive uPN input; mean 5.4, median 5 distinct PNs per KC (matches the 5-7 claw literature) |
| KC <-> APL | all 5,177 KCs both ways | winner-take-all substrate confirmed |
| KC -> MBON | 62,261 | all 96 MBONs (35 types) reached, all KCs contribute |
| DAN -> KC / DAN -> MBON / MBON -> DAN | 47,404 / 2,016 / 2,363 | the reinforcement and feedback paths exist in the data |

Everything the circuit sketch needs is present with the expected statistics.

Male CNS is used only for the demo point cloud unless a sex comparison is
wanted; FlyWire has the mature cell-type labels and the Shiu simulator.

## Forward model (rate-based first, spiking later)

- ORN drive: for each subject, sum over compounds of concentration-scaled DoOR
  response, with a Hill saturation per receptor. Concentration scaling is a
  free parameter; sweep it.
- Antennal lobe: divisive normalisation across glomeruli (Olsen et al. 2010).
- PN to KC: real weights; KC fires if summed input exceeds threshold.
- APL: subtract global inhibition until 5-10% KCs active (winner-take-all).
- KC to MBON: plastic weights, initial value equal per synapse.
- Readout: signed MBON sum with approach/avoid valence from Aso 2014, or
  simply two MBON groups (approach vs avoid). Decision = sign.

This is the Dasgupta 2017 fly hash with real instead of random projection.

## Learning rule

Three-factor: dw_kc_mbon = -lr * KC_active * dopamine, applied only in the
compartment whose DAN fired. Label "cancer" fires the aversive DANs (PPL1),
"control" fires appetitive DANs (PAM). Depression only, plus slow homeostatic
recovery. Springer and Nawrot 2021 (eNeuro, code at github nawrotlab) is the
reference implementation to copy the dynamics from.

## Evaluation (the actual result)

Same feature matrix, same nested 5-fold CV, same seeds, report AUC and
balanced accuracy for Cancer vs Control (drop Benign at first, add as a third
class later):

1. Fly circuit with real PN to KC wiring, three-factor rule.
2. Same circuit with PN to KC replaced by random matrices matched on in-degree
   and out-degree (20 seeds). This is the headline comparison.
3. Same circuit with shuffled labels (permutation null, 200 shuffles).
4. Logistic regression and XGBoost on the 27 raw columns (ceiling for the
   data; Rai 2022 reports ~92% with SVM on 7 VOCs, PLOS One 2026 reports AUC
   0.91 with random forest).
5. Fly circuit fed only the 14 DoOR-measured columns, and only the 13 QSAR
   columns, to show where the signal comes from.

Novelty claim only if (1) beats (2) with a confidence interval that excludes
zero across seeds. If it does not, publish that.

## How to prove the model is right before trusting the classifier

Reproduce known fly biology with the untrained circuit, each as one assert:

- KC sparseness 5-10% for any DoOR odorant.
- KC odor representations are less correlated than PN representations
  (decorrelation, Lin 2014).
- Known innate valences come out with the correct sign from the naive MBON
  readout for a handful of DoOR odorants: apple cider vinegar components and
  ethyl butyrate approach, benzaldehyde and CO2 avoid.
- Classical conditioning: pair odor A with aversive DAN, odor B unpaired; A's
  approach output drops, B's unchanged; generalisation falls with
  PN-similarity; reversal training works.
- Optional stronger check: run the same inputs through the Shiu LIF model
  (Brian2) and confirm MBON spike rates agree in sign with the rate model.

Reproduce the QSAR step honestly: leave-one-odorant-out on DoOR, report per
receptor R, and propagate that error as noise into the classifier runs.

## Demo app (matches demo_videos)

Left pane: 3D fly (NeuroMechFly MuJoCo model, or a static glTF) hovering over
a breath sample vial; a chromatogram-style bar strip of the 27 columns.
Right pane: MaleCNS soma point cloud (from body annotations) with ORN, PN, KC,
MBON somata flashing in sequence as the rate model runs; counters for neurons
fired, KC active fraction, MBON approach vs avoid; verdict banner. Footer
credits: MaleCNS v1.0 CC-BY 4.0, FlyWire v783 CC-BY 4.0, DoOR 2.0.
Stack: Python compute, Three.js viewer, web sockets for frames. Build only
after the evaluation table exists.

## Risks, in order

1. Formula ambiguity and missing DoOR coverage on the cancer markers.
2. Concentration to firing-rate scaling is invented; must be swept and shown
   not to flip the result.
3. Breath panel is one lab, one instrument; no external test cohort exists.
   Cell-line headspace GC-MS papers exist but do not deposit raw data.
4. FlyWire is female, MaleCNS is male; olfactory circuit is isomorphic, fine.
5. 27 features and 350 subjects: XGBoost will likely win outright. The
   interesting claim is only the real-vs-random wiring ablation.

## Order of work

1. Extract circuit matrices, run the biology asserts.
2. Build DoOR join and QSAR fallback, run leave-one-out.
3. Rate model plus learning rule, run the five-arm evaluation.
4. Write the result. Then the demo.
