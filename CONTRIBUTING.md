# Contributing

This project welcomes contributions and suggestions.

## Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA)
declaring that you have the right to, and actually do, grant us the rights to use your
contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply
follow the instructions provided by the bot. You will only need to do this once across
all repositories using our CLA.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional
questions or comments.

## Development setup

```bash
git clone https://github.com/microsoft/BrainCodes.git
cd BrainCodes
pip install -e ".[train]"

git clone https://github.com/klean2050/BioCodec-release.git
pip install -r BioCodec-release/requirements.txt
export PYTHONPATH=$PWD/BioCodec-release:$PYTHONPATH
export BIOCODEC_CKPT=$PWD/BioCodec-release/biocodec_ckpt.pt
```

Verify the install without a dataset:

```bash
python -m training.train --selftest
python examples/quickstart.py
```

## Before opening a pull request

- Format with [black](https://github.com/psf/black): `black braincodes training examples`.
- Keep the released inference path in `braincodes/` free of training-only state;
  pre-training machinery belongs in `training/`.
- `configs/pretrain.yaml` documents the configuration that produced the released
  weights. Do not change its values without re-exporting the checkpoint.

## Reporting security issues

Please do not report security vulnerabilities through public GitHub issues. See
[SECURITY.md](SECURITY.md).
