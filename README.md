# BrainCodes

A compact (1.9M parameters) EEG foundation model that is composed of a [BioCodec](https://arxiv.org/abs/2510.09095) tokenizer and a series of factorized
time/channel transformer layers arranged as an hourglass, with rotary channel
positional encoding. It is pretrained with a masked-code objective on [HBN-EEG](https://neuromechanist.github.io/data/hbn/) releases R3, R4, R7 and R8 (1131 participants). This package ships the pretrained weights plus a minimal usage API.

The backbone is **montage-agnostic**: channel positions enter through a learned
function of 3D electrode coordinates, so you can use any subset/layout of
electrodes (preferably within 10-05) without retraining the encoding.

## Install

```bash
git clone https://github.com/microsoft/BrainCodes.git
cd BrainCodes && pip install -e .
```

### BioCodec tokenizer (required)

BrainCodes operates on discrete codes produced by a **frozen BioCodec** RVQ
tokenizer, which is released separately and is *not* vendored here. Clone it and
install its requirements:

```bash
git clone https://github.com/klean2050/BioCodec-release.git
pip install -r BioCodec-release/requirements.txt
export PYTHONPATH=$PWD/BioCodec-release:$PYTHONPATH
export BIOCODEC_CKPT=$PWD/BioCodec-release/biocodec_ckpt.pt
```

The released checkpoint `biocodec_ckpt.pt` ships with that repository, so the clone
is all you need.

`BIOCODEC_CKPT` is read automatically; alternatively pass the path explicitly:

```python
model = BrainCodesModel.from_pretrained(ch_names, codec_ckpt="/path/to/biocodec_ckpt.pt")
```

The checkpoint is mandatory — without it loading raises rather than silently
falling back to an untrained tokenizer.

## Quickstart

```python
import torch
from braincodes import BrainCodesModel

# Your montage as 10-05 electrode names
ch_names = ["TP9", "AF7", "AF8", "TP10"]
model = BrainCodesModel.from_pretrained(ch_names, device="cuda")

# Raw EEG: (batch, channels, time) at 250 Hz
eeg = torch.randn(8, 4, 1000)

# --- Embeddings --------------------------------------------------------------
emb = model.embed(eeg, pool="channel")   # (B, C*D)  per-channel, flattened
emb = model.embed(eeg, pool="mean")      # (B, D)    mean over channels
grid = model.embed(eeg, pool="none")     # (B, C, T_b, D) bottleneck grid

# --- Linear probe ------------------------------------------------------------
clf = model.build_classifier(num_classes=4, mode="linear_probe")
logits = clf(model.tokenize(eeg))         # (B, 4)
```

See [examples/quickstart.py](examples/quickstart.py) for a complete train loop.

## API

```python
model = BrainCodesModel.from_pretrained(
    ch_names, weights=None, codec_ckpt=None, device="cpu", montage="standard_1005"
)
```
Loads the pretrained weights and binds them to the montage of `ch_names`.
The architecture is read from the checkpoint, so weights stay loadable across
package versions. `codec_ckpt` defaults to `$BIOCODEC_CKPT`.
Create one instance per montage. Then you can use the loaded model to perform either of the following:

```python
codes = model.tokenize(eeg, normalize=True)    # (B, C, T_tok, K)
```
Encode raw EEG `(B, C, T)` (or `(C, T)`) into discrete codes. `normalize=True` z-scores each channel-window.

```python
embeddings = model.embed(eeg, normalize=True, pool="channel")
```
`pool` selects the tap, where `D = 128` is the backbone width:

| `pool`      | shape             | source |
| ----------- | ----------------- | ------ |
| `"channel"` | `(B, C*D)`        | bottleneck, mean over time, flattened |
| `"mean"`    | `(B, D)`          | bottleneck, mean over time and channels |
| `"none"`    | `(B, C, T_b, D)`  | full bottleneck grid |
| `"tokens"`  | `(B, C, T_tok, D)`| full-resolution decoder output |

The first three read the **bottleneck** — the deepest, most time-compressed
encoder representation. This is the tap used for every reported BrainCodes
result. `"tokens"` is the decoder output, which is optimized to reconstruct
codes and therefore retains detail that downstream classifiers generally do not
need; it is exposed for analysis, not recommended as a default.

```python
clf = model.build_classifier(
    num_classes, mode="linear_probe", hidden=256,
    head_type="channel_flat_bottleneck", n_times=None,
)
```
Builds a classification head and applies one of the following freezing strategies:

| `mode`          | trains                              |
| --------------- | ----------------------------------- |
| `linear_probe`  | the head only (backbone frozen)     |
| `full`          | head + full backbone (codec frozen) |

Both heads read the bottleneck:

| `head_type`                            | readout |
| -------------------------------------- | ------- |
| `channel_flat_bottleneck` *(default)*   | per-channel latent → projector → linear |
| `channel_time_flat_bottleneck_noproj`   | flattened bottleneck → linear (strictly linear) |

`n_times` (samples per window) is required for the `noproj` head, whose input
size depends on window length.

The returned module consumes **tokenized codes**: call `clf(model.tokenize(eeg))`.
Under `linear_probe` the backbone is kept in eval mode even if you call
`clf.train()`, so its dropout stays off and the head is fit on exactly the
features it is evaluated on.

## EEG Preprocessing

You can construct the preprocessor from the model requirements:

```python
from braincodes import BrainCodesModel, EEGPreprocessor

model = BrainCodesModel.from_pretrained(["TP9", "AF7", "AF8", "TP10"])
pre = EEGPreprocessor.from_model(model, notch_freq=60.0)

# raw recording at 500 Hz, channels in some other order
eeg = pre(raw, sampling_rate=500, ch_names=raw_ch_names, return_tensors="pt")
emb = model.embed(eeg)
```

Or, construct it directly for full control over the filtering:

```python
pre = EEGPreprocessor(
    ch_names=model.ch_names,
    sampling_rate=250,
    l_freq=0.5,
    h_freq=40.0,
    notch_freq=[60.0, 120.0],
    normalize=False,
)
```

## Pre-training

The full pre-training recipe is included. `configs/pretrain.yaml` is, value for
value, the configuration that produced `weights/braincodes.pt`.

The training code is not part of the installed package; run it from the repo
root after installing its extra requirement:

```bash
pip install -e ".[train]"
```

```bash
# 1. Build a tokenized cache from an HBN-EEG BIDS release (once per release).
python -m training.preprocess_hbn \
    --bids_root data/HBN-EEG/R4 --out_dir data/cache/R4 \
    --keep_subset gsn128_to_64 --keep_raw \
    --target_sr 250 --segment_seconds 30 --device cuda

# 2. Point configs/pretrain.yaml at the manifests it writes, then train.
python -m training.train --config configs/pretrain.yaml

# Architecture-only smoke test, no dataset needed:
python -m training.train --selftest
```

The released checkpoint is 20k steps on HBN-EEG releases R3, R4, R7 and R8
(4x A6000, ~2.75 h per 1k steps). The objective is masked-code reconstruction
with contiguous spatio-temporal spans, plus an oscillation-injection auxiliary
that predicts the frequency, duration and location of synthetic bursts.

Training-only machinery (masking, the masked-code heads, the injection
auxiliaries) lives in `training/pretrain_model.py` as a subclass, so the
released inference path in `braincodes/` carries no training state. Export an
inference checkpoint from a pre-training checkpoint with:

```python
model.export_release_weights("weights/braincodes.pt", step=20000)
```

The cached shards hold only signal and electrode coordinates. Participant
identifiers stay in `manifest.json`, where they are needed for the leak-free
subject-level split; demographics and phenotype scores are never read or written.

## Structure

```
braincodes/
    api.py          BrainCodesModel: load / tokenize / embed / build heads
    backbone.py     BrainCodes backbone + frozen BioCodec tokenizer
    classifier.py   classification head, freezing strategies
    channels.py     electrode names -> coordinates (via MNE montages)
    preprocess.py   EEGPreprocessor: resample / filter / channel-align
training/
    train.py            pre-training loop and entry point
    pretrain_model.py   masking + masked-code / injection heads
    preprocess_hbn.py   HBN-EEG BIDS -> tokenized cache
    loader.py           manifests, subject split, dataloaders
    hbn_dataset.py      cached-shard dataset
    multi_length.py     variable-window sampling
    inject.py           synthetic oscillation bursts + EEG band edges
    objective.py        masked token CE + injection losses
configs/
    pretrain.yaml       the released pre-training configuration
examples/
    quickstart.py   embeddings + a linear probe, end to end
docs/               project page (published via GitHub Pages)
weights/
    braincodes.pt   pretrained encoder weights (1.9M params, 7.2 MB)
```

The BioCodec tokenizer is **not** included; see [Install](#biocodec-tokenizer-required).

## Privacy

This package does not collect, transmit, or store any data. All processing is local.
See the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).

## Intended Use

BrainCodes is released for neuroscientific and machine-learning research. It is **not** a
medical device and must not be used for clinical diagnosis, treatment decisions, biometric
identification, or inferences about an individual's mental state or ability. See
[MODEL_CARD.md](MODEL_CARD.md) for the full intended-use statement and limitations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Citation

See [CITATION.cff](CITATION.cff). Please cite BrainCodes together with the Healthy Brain
Network datasets it was pretrained on — see [NOTICE.md](NOTICE.md) for the required
attributions.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE). Third-party
components (BioCodec, EnCodec) and the training-data attribution requirements are
documented in [NOTICE.md](NOTICE.md).

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause
confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is
subject to those third parties' policies.


