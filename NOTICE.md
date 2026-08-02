# Third-Party Notices

This repository incorporates material from the projects listed below. The original
copyright notices and the licenses under which Microsoft received such material are set
forth here for informational purposes only. Microsoft licenses its own contributions
under the terms in [LICENSE](LICENSE).

## 1. BioCodec

This repository does **not** redistribute BioCodec source code or tokenizer weights;
they are obtained separately from the BioCodec release (see the README). BrainCodes
depends on BioCodec at runtime, and `weights/braincodes.pt` contains code-embedding
tables that were *initialized* from the BioCodec codebooks before being trained, so the
released weights are a derived work of BioCodec.

Source: "Neural Codecs as Biosignal Tokenizers"

```
MIT License

Copyright (c) 2026 Kleanthis Avramidis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. EnCodec (Meta Platforms, Inc.)

BioCodec derives from Meta's EnCodec (SEANet encoder/decoder, normalized convolutions,
residual vector quantization). No EnCodec-derived code is distributed in this repository;
this notice is retained because the BioCodec-initialized code embeddings in
`weights/braincodes.pt` trace back to that lineage.

Source: https://github.com/facebookresearch/encodec

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Healthy Brain Network EEG (training data)

The pretrained weights in `weights/` were derived from the HBN-EEG BIDS releases R3, R4,
R7 and R8, distributed by the Child Mind Institute under
[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Only participants flagged
`commercial_use = Yes` are included; the Not-for-Commercial-Use cohort (`cmi_bids_NC`) was
not used.

Attribution required by the dataset:

- Alexander LM, et al. An open resource for transdiagnostic research in pediatric mental
  health and learning disorders. *Scientific Data* 4:170181 (2017).
  https://dx.doi.org/10.1038/sdata.2017.181
- O'Connor D, et al. The Healthy Brain Network Serial Scanning Initiative.
  *Scientific Data* 4:170040 (2017). https://dx.doi.org/10.1038/sdata.2017.40
- Shirazi SY, et al. HBN-EEG: The FAIR implementation of the Healthy Brain Network EEG
  dataset (2024). https://doi.org/10.1101/2024.10.03.615261
