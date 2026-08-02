"""Minimal end-to-end example: embeddings + a linear probe.

Requires the BioCodec tokenizer checkpoint (see the README). Run from the repo
root after ``pip install -e .``::

    export BIOCODEC_CKPT=/path/to/biocodec_ckpt.pt
    python examples/quickstart.py

For full fine-tuning use ``mode="full"``. For a strictly linear probe use
``head_type="channel_time_flat_bottleneck_noproj", n_times=n_times``.
"""

import torch

from braincodes import BrainCodesModel

ch_names = ["TP9", "AF7", "AF8", "TP10"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BrainCodesModel.from_pretrained(ch_names, device=device)

n_times = 1000  # 4 s
eeg = torch.randn(8, len(ch_names), n_times)

print("channel embeddings:", tuple(model.embed(eeg, pool="channel").shape))
print("mean embedding:    ", tuple(model.embed(eeg, pool="mean").shape))

clf = model.build_classifier(num_classes=3, mode="linear_probe")
opt = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=5e-4)
labels = torch.randint(0, 3, (eeg.shape[0],), device=device)

codes = model.tokenize(eeg)  # (B, C, T_tok, K) — tokenize once, it is frozen
clf.train()
for step in range(5):
    loss = torch.nn.functional.cross_entropy(clf(codes), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {step}  loss {loss.item():.3f}")
