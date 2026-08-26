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

# BrainCodes

BrainCodes is a compact (~1.9M parameter) EEG foundation model that operates over discrete
tokens rather than raw waveforms. A frozen [BioCodec](https://arxiv.org/abs/2510.09095)
residual vector quantizer converts each EEG channel into codebook indices, and a factorized
time/channel transformer is pretrained over those codes with a masked-code objective.

- **Developed by:** Microsoft Research
- **Model type:** Self-supervised encoder for multi-channel EEG
- **Input:** Raw EEG `(batch, channels, time)` at 250 Hz
- **Output:** Discrete codes `(B, C, T_tok, K)` or embeddings of width `D = 128`
- **Repository:** https://github.com/microsoft/BrainCodes
- **Paper:** in preparation; this card will be updated with the reference on publication.
- **License:** MIT (see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md))

## Architecture

| Component | Setting |
| --- | --- |
| Tokenizer | BioCodec RVQ, frozen, causal, 250 Hz, `K = 6` codebooks, 256 entries each |
| Token width | `d_model = 16` per codebook, concatenated to `96` and projected to the backbone width `D = 128` |
| Backbone | Hourglass of 3 blocks (pool factors 2, 2), each 2 time-attention + 1 channel-attention layer, RoPE time positions |
| Channel conditioning | Rotary positional encoding over 3D electrode coordinates, re-applied at every block |
| Parameters | ~1.9M trainable (codec frozen and not redistributed) |
| Checkpoint | `weights/braincodes.pt` (7.2 MB, encoder only; BioCodec installed separately) |

The backbone is **montage-agnostic**: electrode identity enters only through 3D coordinates,
so any subset or layout of 10-05 electrodes works without retraining the channel encoding.

## Training Data

Pretrained on the [HBN-EEG](https://neuromechanist.github.io/data/hbn/) BIDS releases
**R3, R4, R7, and R8** — 1131 unique participants of the Child Mind Institute Healthy Brain
Network, a publicly released, de-identified developmental EEG cohort (ages 5-21).

All 1131 participants are flagged `commercial_use = Yes` in the release `participants.tsv`.
The Not-for-Commercial-Use cohort (`cmi_bids_NC`) is distributed as a separate dataset and was
not used. Phenotypic data was not used, so no HBN Data Usage Agreement was required.

These releases are distributed under **CC-BY-SA 4.0**. The released weights are not an
adaptation of the dataset, so the ShareAlike term does not extend to them; the weights are
licensed under the MIT License in [LICENSE](LICENSE).

Per the dataset's acknowledgement requirements, work using this model should also cite
[Alexander et al. 2017](https://dx.doi.org/10.1038/sdata.2017.181),
[the HBN protocol paper](https://dx.doi.org/10.1038/sdata.2017.40), and
[the HBN-EEG release](https://doi.org/10.1101/2024.10.03.615261).

The release contains model parameters only -- no recordings, segments, codes or participant
identifiers. We have not run a membership-inference or reconstruction attack against the
checkpoint, so we make no formal privacy guarantee. Note also that the learned
representations are strongly subject-discriminative, as is typical for EEG encoders: a
linear probe separates individuals within a cohort at high accuracy. Treat embeddings
derived from identifiable recordings as personal data.

## Intended Use

**Intended:** neuroscientific and machine-learning research — representation learning for EEG,
linear probing and fine-tuning on downstream classification tasks, benchmarking against other
EEG encoders, and on-device/low-compute experimentation.

**Out of scope:** BrainCodes is **not** a medical device and must not be used for clinical
diagnosis, treatment decisions, seizure or pathology detection in care settings, employment or
insurance screening, biometric identification, or any inference about mental state, emotion, or
cognitive ability about an individual. It has not been validated for any of these purposes and
has no regulatory clearance.

## Limitations

- Pretraining data is developmental EEG from a single cohort; performance on adult, clinical,
  or low-density consumer montages is untested and may degrade.
- The tokenizer is causal and fixed at 250 Hz. Inputs at other rates must be resampled with
  `EEGPreprocessor`; artifacts from aggressive resampling propagate into the codes.
- Normalization at inference must match pretraining. `normalize=True` applies per-channel-window
  z-scoring; using a different scheme silently shifts the code distribution and degrades
  embeddings.
- The model has no notion of artifact rejection. Eye blinks, EMG, and electrode pops are
  tokenized like any other signal.
- Evaluation to date is on a limited set of public benchmarks; results have not been audited for
  fairness across demographic subgroups.
- Embeddings may retain subject-identifying structure and should not be shared as if they were
  anonymized.

## Evaluation

Benchmark results are reported in the accompanying paper and are not reproduced here ahead of
publication. The protocol is: subject-level (leak-free) train/val/test splits, three seeds per
configuration, and balanced accuracy reported as mean ± standard deviation; multi-class tasks
additionally report Cohen's kappa and weighted F1, and binary tasks report AUROC and AUC-PR.
This card will be updated with the full table on publication.

## Usage

```python
import torch
from braincodes import BrainCodesModel

ch_names = ["TP9", "AF7", "AF8", "TP10"]
model = BrainCodesModel.from_pretrained(ch_names, device="cuda")

eeg = torch.randn(8, 4, 1000)              # (B, C, T) @ 250 Hz
emb = model.embed(eeg, pool="channel")     # (B, C*D)

clf = model.build_classifier(num_classes=4, mode="linear_probe")
logits = clf(model.tokenize(eeg))
```

## Third-Party Components

This model requires the BioCodec encoder and residual vector quantizer, which itself
derives from Meta's EnCodec. BioCodec is **not** bundled here: install it separately as
described in the [README](README.md). Both are MIT licensed; full notices are in
[NOTICE.md](NOTICE.md).

## Privacy

This model performs all processing locally and does not collect, transmit, or store any data.
See the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).

## Citation

```bibtex
@article{braincodes2026,
  title   = {BrainCodes: Toward Lightweight EEG Foundation Models},
  author  = {Avramidis, Kleanthis and Emmanouilidou, Dimitra and Tashev, Ivan},
  year    = {2026}
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use
of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause
confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject
to those third parties' policies.
