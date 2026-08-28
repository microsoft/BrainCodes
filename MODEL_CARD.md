---
license: mit
license_link: LICENSE
library_name: braincodes
pipeline_tag: feature-extraction
tags:
  - eeg
  - biosignals
  - foundation-model
  - representation-learning
  - neural-codec
---

# BrainCodes: Toward Lightweight EEG Foundation Models

**Last Updated:** 28-AUG-2026

# Model summary

| Developer | Microsoft Corporation |
|----|----|
| Developed by | Microsoft Research |
| Description | A compact self-supervised encoder for multi-channel EEG. A quantization-based tokenizer converts each EEG channel into discrete code indices, and a factorized time/channel transformer is pretrained over those codes with masked-code prediction and synthetic oscillation-injection objectives. The model outputs embeddings for downstream research use; it ships no trained task head. It was pretrained on publicly released, de-identified developmental EEG from the Healthy Brain Network. |
| Model architecture | Factorized time/channel transformer with rotary spatiotemporal position embeddings over discrete codec tokens; frozen RVQ tokenizer front-end. |
| Parameters | Approximately 1.9M trainable parameters (BioCodec tokenizer frozen and excluded) |
| Inputs | Other modality: multi-channel EEG time series at 250 Hz. No text, image, audio, or video input. Channel names must resolve within the selected MNE standard montage. Pretraining used windows of 2–30 s with random subsets of 16–64 channels. The public API accepts variable input duration and channel count, but configurations outside the pretraining ranges have not been validated. |
| Outputs | Other modality: `tokenize()` returns discrete code indices with shape `(batch, channels, token time, codebooks)`. `embed()` returns continuous backbone representations with feature width 128. With `pool="channel"`, per-channel bottleneck representations are flattened to `(batch, channels × 128)`; `pool="mean"` returns `(batch, 128)`; `pool="none"` returns the bottleneck grid `(batch, channels, bottleneck time, 128)`; and `pool="tokens"` returns `(batch, channels, token time, 128)`. |
| Context length | Approximately 20.8 tokens per second per channel, measured as 625 tokens per channel for a 30 s window. Pretraining used up to 30 s per window. |
| Training dates | June 2026 to July 2026 |
| Release date | Estimated for September 2026 |
| Repository | https://github.com/microsoft/BrainCodes |
| Paper | In preparation; this model card will be updated with the publication reference when available. |
| License | Free/open source — MIT License |
| Model dependencies | [BioCodec](https://arxiv.org/abs/2510.09095), a frozen residual vector quantizer derived from an open-source version of EnCodec. BioCodec is not bundled with BrainCodes and must be installed separately as described in the README. Both components are MIT licensed; attribution notices are provided in NOTICE.md. |
| List and link to any additional related assets | Public GitHub repository containing inference code, model card, and usage examples, under the MIT License. Training data is the publicly available HBN-EEG BIDS release (CC-BY-SA 4.0). Third-party notices in NOTICE.md. |
| Acceptable use policy | N/A. Intended and out-of-scope uses are stated in Sections 2.1 and 2.2. |

# 1. Model overview

The model is a compact EEG encoder trained in two stages. A residual vector quantizer, pretrained separately and held frozen, converts each EEG channel into sequences of discrete code indices drawn from 6 codebooks of 256 entries each. A transformer backbone, consisting of 2 time-attention layers and 1 channel-attention layer per block, with rotary spatiotemporal position embeddings, is then pretrained over those codes using a masked-code objective, in which spans of codes are masked and the model predicts the missing indices. Pretraining also includes synthetic transient narrowband oscillation injection, with auxiliary objectives that predict the injected oscillation’s frequency, duration, and location. The transformer never observes the raw waveform; it operates entirely on integer code IDs. The backbone contains approximately 1.9M trainable parameters; the BioCodec tokenizer is frozen and excluded from this count.

Two design properties emphasized in this release are its compact size and montage flexibility. First, at approximately 1.9M trainable parameters, it is smaller than each comparison model listed in Section 3, making it a candidate for low-compute experimentation. Second, it is montage-flexible: electrode identity enters the model through 3D coordinates. The public API accepts user-selected subsets of electrode labels present in a selected MNE standard montage, allowing a model instance to be configured for different supported channel sets without retraining the channel encoding.

## 1.1 Architecture details

| Component | Setting |
|---|---|
| Tokenizer | BioCodec RVQ, frozen and causal, operating at 250 Hz with 6 codebooks of 256 entries each |
| Token width | 16 dimensions per codebook, concatenated to 96 dimensions and projected to the 128-dimensional backbone width |
| Backbone | Three-block hourglass with pooling factors of 2 and 2; each block contains two time-attention layers and one channel-attention layer |
| Position encoding | Rotary time-position encoding and rotary spatial encoding over 3D electrode coordinates, applied in every block |
| Pretraining variation | Window lengths of 2, 5, 10, 15, 20, and 30 seconds; random subsets of 16–64 channels; electrode-coordinate jitter; masked code spans; and transient narrowband oscillation injection |
| Parameters | 1,877,312 trainable parameters, approximately 1.9M; BioCodec is frozen and excluded |
| Checkpoint | `weights/braincodes.pt`, 7.19 MiB, encoder only; BioCodec is installed separately |

## 1.2 Key model capabilities

- Embedding extraction from variable-length multi-channel EEG, with a 128-dimensional feature width and multiple pooling modes

- Montage-flexible operation on electrode subsets resolved from a selected MNE standard montage using 3D coordinate embeddings

- Support for linear probing and full fine-tuning on user-supplied labeled datasets

- Compact architecture intended for low-compute research on a CPU or single GPU; on-device performance has not been benchmarked

# 2. Usage

## 2.1 Primary use cases

Intended for neuroscientific and machine-learning research: representation learning for EEG, linear probing and fine-tuning on downstream classification tasks, benchmarking against other EEG encoders, and low-compute experimentation.

The model produces representations only. To obtain any prediction, a downstream researcher must supply their own labeled dataset and train a classification head on top of the embeddings. No trained task head is distributed.

## 2.2 Out-of-scope use cases

This model is not a medical device and must not be used for clinical diagnosis, treatment decisions, or seizure or pathology detection in care settings. It must not be used for employment or insurance screening, biometric identification, or consequential decisions based on inferred mental state, emotion, or cognitive ability. It has not been validated for any of these purposes and holds no regulatory clearance. Any investigation of downstream clinical or consequential applications falls outside this release’s intended use and would require independent validation, scientific evidence, and regulatory clearance, all of which remain the responsibility of the investigating party.

## 2.3 Distribution channels

BrainCodes distribution is as follows:

- Public access through an open-source GitHub repository, including model weights and usage examples.

- The model is not distributed through any API, enterprise offering, subscription service, or Microsoft-hosted endpoint.

## 2.4 Technical requirements and integration guidance

Requires Python 3.10 or later, PyTorch 2.0 or later, NumPy 1.23 or later, SciPy 1.9 or later, MNE 1.5 or later, and the separately installed BioCodec dependency. The software supports execution on a CPU or a single GPU and does not require distributed or specialized hardware; runtime performance has not been benchmarked. Integration is via the Python package, which exposes methods to tokenize EEG, extract embeddings, and construct classification heads. Installation and integration guidance is provided in the repository README.

Input EEG must be channel-aligned and sampled or resampled to 250 Hz before use. Band-pass filtering consistent with the intended preprocessing pipeline is recommended; `EEGPreprocessor` provides these operations. By default, `EEGPreprocessor` uses a 0.5–40 Hz band-pass filter, a 60 Hz notch filter, and `normalize=False`, whereas pretraining used a 0.5–100 Hz band-pass filter with 60 Hz and 120 Hz notch filters when applicable. Users seeking closer preprocessing parity should use `h_freq=100.0` and, where applicable, `notch_freq=[60.0, 120.0]`. In `BrainCodesModel.tokenize()` and `BrainCodesModel.embed()`, the default `normalize=True` applies mean and standard-deviation normalization independently to each channel within each input window. Pretraining preprocessing instead used per-channel median and median-absolute-deviation normalization before segmentation, with normalized values limited to the range −20 to 20. Users who reproduce the pretraining normalization externally should pass `normalize=False` to avoid applying a second normalization step. Differences in preprocessing can shift the discrete-code distribution and affect embedding quality.

The model may be integrated into research analysis pipelines and human-supervised, non-clinical research prototypes. It is **not** suitable for integration into autonomous applications, real-time clinical monitoring, or any system making consequential decisions about individuals.

### 2.4.1 Basic usage

After installing BrainCodes and the separate BioCodec dependency and configuring the BioCodec checkpoint:

```python
import torch
from braincodes import BrainCodesModel

channel_names = ["TP9", "AF7", "AF8", "TP10"]
model = BrainCodesModel.from_pretrained(channel_names, device="cpu")

eeg = torch.randn(8, 4, 1000)              # (batch, channels, time) at 250 Hz
embeddings = model.embed(eeg, pool="channel")

classifier = model.build_classifier(num_classes=4, mode="linear_probe")  # newly initialized; train on labeled data
logits = classifier(model.tokenize(eeg))
```

## 2.5 Supported languages

Not applicable. The model does not process natural language; it accepts EEG time-series input only. Documentation is provided in English.

## 2.6 Responsible AI considerations

Developers building on top of this model should treat EEG as sensitive personal data and apply appropriate consent, governance, and data-protection controls, including applicable human-subjects research approvals. The released library performs processing locally and does not itself collect, transmit, or store input data. Embeddings may retain subject-identifying structure and should not be shared as if they were anonymized. Treat embeddings derived from identifiable recordings as personal data. Because the model outputs only representations, the party training any downstream classification head is responsible for evaluating the validity, fairness, and safety of the resulting predictions. Health-adjacent and consequential applications are outside the intended scope; this release provides neither independent clinical validation nor regulatory clearance. Developers should retain meaningful human oversight and must not present outputs as diagnostic. Results have not been audited for fairness across demographic subgroups. For additional information, see the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).

## 2.7 Known limitations

- Pretraining data is developmental EEG (ages 5–21) from a single but large cohort. Performance on adult, clinical, underrepresented populations or low-density consumer montages is therefore untested and may degrade or be unstable.

- The tokenizer is causal and fixed at 250 Hz; artifacts from aggressive resampling or incorrect or inappropriate preprocessing directly propagate into the codes.

- The model has no artifact rejection. Eye blinks, muscle activity, and electrode pops are tokenized like any other signal and may influence the resulting tokens and embeddings.

- Evaluation to date is limited to five public classification benchmarks spanning event classification, motor imagery, cognitive-state classification, and sleep staging; generalization to other tasks remains untested.

# 3. Quality and performance evaluation

BrainCodes was evaluated with task-specific downstream heads on five public classification benchmarks spanning motor imagery (BCIC-IV-2a and PhysioNet-MI), event classification (TUEV), cognitive-state classification (EEGMAT), and sleep staging (EESM23). The BrainCodes row reports balanced accuracy as the mean ± standard deviation across three random seeds; the other rows contain reported comparison results and were not produced by this repository.

| **Model** | **TUEV** | **BCIC-IV-2a** | **EEGMAT** | **PhysioNet-MI** | **EESM23** |
|----|----|----|----|----|----|
| [BIOT](https://arxiv.org/abs/2305.10351) | 0.528 ± 0.023 | 0.475 ± 0.009 | 0.688 ± 0.019 | 0.370 ± 0.032 | 0.386 ± 0.009 |
| [LaBraM-Base](https://arxiv.org/abs/2405.18765) | 0.641 ± 0.007 | 0.487 ± 0.009 | 0.691 ± 0.013 | 0.372 ± 0.046 | 0.389 ± 0.018 |
| [CBraMod](https://arxiv.org/abs/2412.07236) | 0.667 ± 0.011 | 0.514 ± 0.007 | 0.726 ± 0.013 | 0.385 ± 0.035 | 0.434 ± 0.021 |
| [REVE-Base](https://arxiv.org/abs/2510.21585) | <u>0.676 ± 0.023</u> | **0.640 ± 0.010** | <u>0.766 ± 0.036</u> | **0.537 ± 0.005** | <u>0.455 ± 0.025</u> |
| **BrainCodes** | **0.684 ± 0.005** | <u>0.617 ± 0.007</u> | **0.770 ± 0.041** | <u>0.456 ± 0.023</u> | **0.482 ± 0.036** |

The cited model publications are Yang, Westover, and Sun, *BIOT: Biosignal Transformer for Cross-Data Learning in the Wild* (NeurIPS 2023); Jiang, Zhao, and Lu, *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI* (ICLR 2024); Wang et al., *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding* (ICLR 2025); and El Ouahidi et al., *REVE: A Foundation Model for EEG Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects* (2025). The TUEV, BCIC-IV-2a, EEGMAT (identified as MAT), and frozen-encoder PhysioNet-MI comparison values are traceable to the REVE study's reported-results and frozen-encoder tables. The EESM23 comparison values do not appear in those cited model papers and should be treated as unverified comparison data pending a separate provenance record.

Among the models listed in the comparison table, BrainCodes has the highest mean balanced accuracy on TUEV, EEGMAT, and EESM23, and the second-highest on BCIC-IV-2a and PhysioNet-MI. It does so with approximately 1.9M trainable parameters, compared with the published encoder sizes of 3.2M for BIOT, 5.8M for LaBraM-Base, 4.0M for CBraMod, and 69M for REVE-Base. Results remain subject to change as downstream optimization is refined.

Bias-related evaluation safeguards include subject-disjoint train and test partitions, class-weighted training on all imbalanced tasks, and balanced accuracy as the sole model-selection and reporting metric to reduce the risk of majority-class performance being mistaken for overall competence.

## 3.1 Benchmarking methodology

Each benchmark dataset was band-pass filtered, notch filtered where mains interference was present, resampled to 250 Hz, and passed through the frozen tokenizer before reaching the model. Most benchmarks used full fine-tuning of the pretrained encoder with a bottleneck classification head. PhysioNet-MI used a linear probe over the encoder and was therefore compared against frozen-encoder results rather than published fine-tuned results. Balanced accuracy was the primary metric and the sole model-selection metric. Checkpoints were chosen using validation balanced accuracy, class weights were enabled for every task, and no threshold tuning was applied.

- **TUEV (TUH EEG Events)** is a six-way classification of annotated events including spike-and-slow-wave, eye movement, and artifact, evaluated over a 16-channel bipolar double-banana montage. Reference: Obeid, I. & Picone, J. The Temple University Hospital EEG Data Corpus. *Frontiers in Neuroscience* 10:196, 2016.

- **BCIC-IV-2a (BCI Competition IV, data set 2a)** is a four-way classification of motor imagery of the left hand, right hand, both feet, and tongue, evaluated over a 22-channel montage across nine subjects and two recording sessions. Reference: Tangermann, M. et al. Review of the BCI Competition IV. Frontiers in Neuroscience 6:55, 2012.

- **EEGMAT (Mental Arithmetic Task)** is a binary classification of resting state versus serial-subtraction mental arithmetic, evaluated over a 20-channel montage across 36 subjects with one resting-state recording and one mental-arithmetic recording each. Reference: Zyma, I. et al. Electroencephalograms during Mental Arithmetic Task Performance. Data 4(1):14, 2019.

- **PhysioNet-MI (EEG Motor Movement/Imagery Database)** is a four-way classification of imagined movement of the left fist, right fist, both fists, and both feet, evaluated over a 64-channel montage across 109 subjects. Reference: Schalk, G. et al. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034–1043, 2004.

- **EESM23 (Ear-EEG Sleep Monitoring 2023, OpenNeuro)** is a six-way classification of sleep stage covering wake, REM, N1, N2, N3, and artifact, evaluated over a 4-channel in-ear montage across ten subjects. Reference: Rezaei Tabar, Y., Mikkelsen, K., Kidmose, P. et al. EESM23. Scientific Data 12:301, 2025.

## 3.2 Safety evaluation and red-teaming

The standard generative-content risk categories — sexual, violent, hateful, or self-harm content, copyright and IP reproduction, and jailbreaks — are **not applicable**. The model accepts EEG signals and emits numeric embeddings; it has no natural-language or media interface and cannot produce such content or content that could be interpreted as such.

Safety review focused on risks relevant to this modality. Checkpoint contents: the released checkpoint contains only model parameters—no EEG recordings, signal segments, discrete codes, or participant identifiers. Data rights: the HBN release metadata records for all 1,131 pretraining participants had `commercial_use = Yes`, and the separately distributed Not-for-Commercial-Use cohort was excluded. Misuse: the primary identified risk is downstream clinical application, mitigated by shipping no trained task head, by explicit out-of-scope documentation, and by the requirement that users control their own data.

We have not run a membership-inference or reconstruction attack against the checkpoint, so we make no formal privacy guarantee.

# 4. Data overview

## 4.1 Training, testing, and validation datasets

Pretraining used the publicly released HBN-EEG BIDS dataset from the Child Mind Institute Healthy Brain Network, a de-identified developmental EEG cohort of participants aged 5–21. Recordings covered resting state and a set of passive-viewing and active cognitive tasks. Recordings were segmented into 30-second windows; 20% of participants were held out for subject-disjoint validation. The model was trained on released shards R3, R4, R7, and R8, which, after bad-signal rejection, included 1,131 participants and 98,401 30-second, 64-channel EEG windows in total. The five downstream evaluation datasets are described in Section 3.1.

The HBN release metadata records for all 1,131 participants used for pretraining had `commercial_use = Yes`. The separately distributed Not-for-Commercial-Use cohort (`cmi_bids_NC`) was not used. Phenotypic data was also not used, so no HBN Data Usage Agreement was required.

**4.1.1 Size of dataset and characteristics**

**4.1.2.A Text training data size:** Not applicable. Text data is not part of the training data.

**4.1.2.B Text training data content:** Not applicable.

**4.1.2.C Image training data size:** Not applicable. Images are not part of the training data.

**4.1.2.D Image training data content:** Not applicable.

**4.1.3.A Audio training data size:** Not applicable. Audio data is not part of the training data.

**4.1.3.B Audio training data content:** Not applicable.

**4.1.4.A Video training data size:** Not applicable. Video data is not part of the training data.

**4.1.4.B Video training data content:** Not applicable.

**4.1.5.A Other training data size:** Biosignal (electroencephalography – HBN EEG): 1,131 participants and 98,401 30-second, 64-channel EEG windows (about 820 recording hours in total).

**4.1.5.B Other training data content:** De-identified scalp EEG recorded during resting state and standardized passive-viewing and active cognitive tasks, from a publicly released developmental research cohort, available at https://neuromechanist.github.io/data/hbn/.

**4.1.6 Latest date of data acquisition/collection for model training:** 29-May-2026

**4.1.7 Is data collection ongoing to update the model with new data collection after deployment?** No.

**4.1.8 Date the training dataset was first used to train the model:** June 2026

**4.1.9 Rationale or purpose of data selection:**

HBN-EEG was selected as the most appropriate publicly available source of large-scale, montage-consistent EEG with permissive redistribution terms, allowing replication of scientific findings. The dataset spans developmental ages with substantial physiological variability, provides high-density recordings that support spatial modeling of electrode positions, and includes multiple passive and active recordings per participant.

## 4.2 List of data sources

### 4.2.1 Publicly available datasets

The pretraining source was the [HBN-EEG BIDS dataset](https://neuromechanist.github.io/data/hbn/), specifically releases R3, R4, R7, and R8. These releases are distributed under CC BY-SA 4.0.

Per the dataset’s acknowledgement requirements, work using BrainCodes should cite:

- [Alexander et al. (2017), *An open resource for transdiagnostic research in pediatric mental health and learning disorders*](https://doi.org/10.1038/sdata.2017.181)
- [Healthy Brain Network protocol paper](https://doi.org/10.1038/sdata.2017.40)
- [HBN-EEG release](https://doi.org/10.1101/2024.10.03.615261)

**4.2.1.1 Have you used publicly available datasets to train the model?** Yes

### 4.2.2 Private non-publicly available datasets obtained from third parties

#### 4.2.2.1 Datasets commercially licensed by rightsholders or their representatives

**4.2.2.1A Have you concluded transactional commercial licensing agreement(s) with rightsholder(s) or with their representatives?** Not applicable.

#### 4.2.2.2 Private datasets obtained from other third parties

**4.2.2.2.A Have you obtained private datasets from third parties that are not licensed as described in Section 4.2.2.1, such as data obtained from providers of private databases, or data intermediaries?** No.

## 4.3 Personal Data

**4.3.1 Was personal data used to train the model?** Pretraining used publicly released, de-identified EEG recordings. No direct participant identifiers or phenotypic data were used. Microsoft follows applicable laws and best practices pertaining to personal data.

## 4.4 Synthetic data

**4.4.1 Was any synthetic AI-generated data used to train the model?** No synthetic AI-generated EEG recordings were used as source data. Algorithmically generated narrowband oscillations were injected into real EEG as a training augmentation for auxiliary prediction objectives.

## 4.5 Data processing aspects

### 4.5.1 Respect of reservation of rights from text and data mining exception or limitation

**4.5.1A Does this dataset include any data protected by copyright, trademark, or patent?** Yes. The HBN-EEG BIDS releases used for pretraining are distributed under CC BY-SA 4.0 and were used subject to that license’s terms. Microsoft follows applicable laws and best practices for processing data protected by copyright, trademark, or patent.

### 4.5.2 Other information

**4.5.2A Does the dataset include information about consumer groups without revealing individual consumer identities?** No. HBN-EEG is a de-identified research cohort, and no consumer-group information was used for pretraining. Microsoft follows applicable laws and best practices for protecting consumer identities.

**4.5.2B Was the dataset cleaned or modified before model training?** Yes. EEG channels were selected, band-pass filtered from 0.5 to 100 Hz, notch filtered at 60 and 120 Hz when applicable, and resampled to 250 Hz. Flat channels were detected and spline-interpolated; recordings with more than half of their channels detected as flat were excluded. Signals were robustly normalized per channel using the median and median absolute deviation. Normalized values were then limited to the range −20 to 20 to bound extreme amplitudes. Signals were segmented into 30-second windows and tokenized with the frozen BioCodec tokenizer.

# 5. Contact

Requests for additional information can be directed to <MSFTAIActRequest@microsoft.com>.

**Authorized representative:** Microsoft Ireland Operations Limited, 70 Sir John Rogerson’s Quay, Dublin 2, D02 R296, Ireland

# 6. Appendix

The pretraining objective is masked discrete-code prediction over six codebooks with 256 entries each, so chance top-1 accuracy for each codebook is 0.39%. The table reports four code-prediction measurements used to assess temporal and cross-channel structure.

| **Experiment** | **BrainCodes** |
|----|----|
| Top-1 code prediction, no anchors | **10.7%** |
| Baseline of copying the previous code | 1.4% |
| Masked row prediction given 10% of codes and no context | **1.5%** |
| Masked row prediction given 10% of codes and channel context | **17.7%** |

Unanchored code prediction reaches 27× chance and exceeds the 1.4% previous-code baseline, providing evidence beyond simple persistence. With 10% of codes and no channel context, masked-row prediction reaches 1.5%, which is comparable to the 1.4% previous-code baseline. Supplying the other channels at the same point lifts the task to 17.7%, a 12× gain. These results provide evidence that the backbone learns cross-channel and temporal structure; downstream task performance is reported separately in Section 3. They should not be interpreted as a formal privacy evaluation; no membership-inference or reconstruction attack has been performed.

# 7. Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third parties’ policies.
