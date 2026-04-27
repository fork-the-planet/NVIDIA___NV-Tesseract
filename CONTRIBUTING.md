# Contributing to NV-Tesseract

Thank you for your interest in this project.

## Licensing

By contributing, you agree that your contributions will be licensed under the **Apache License, Version 2.0**, the same license that covers the project (see [`LICENSE`](LICENSE) in the repository root).

Third-party notices and PyPI dependency summaries are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). **Verbatim upstream license files** for source included in this repo live under [`third_party/`](third_party/) (see [`third_party/README.md`](third_party/README.md)).

### Source file headers

- **NVIDIA-authored** Python files use SPDX short-form tags (`SPDX-FileCopyrightText`, `SPDX-License-Identifier`) per SPDX Specification v2.3, Annex E.
- Files that include third-party or modified third-party code must retain **upstream copyright and license notices** in the file (SPDX alone is not a substitute where upstream terms require full notice).

You can refresh SPDX headers on Python trees with:

```bash
make spdx
```

Verify headers before submitting:

```bash
make spdx-check
```

## Intellectual property review (NVIDIA contributors)

If you are contributing as part of your work at NVIDIA, follow your organization’s **IP Review Process for Open Source** (internal documentation on Confluence: *IP Review Process — Open Source*) before submitting changes intended for external release. Ensure your contributions are cleared for distribution under Apache-2.0 and do not incorporate material you are not entitled to license.

External contributors should ensure they have the rights to submit their changes under Apache-2.0 and that third-party code meets the notice requirements above.

## Developer Certificate of Origin

We use the **Developer Certificate of Origin (DCO)** for contribution approval. The canonical text is published at:

**[https://developercertificate.org/](https://developercertificate.org/)**

### Signing your work

Every commit must include a `Signed-off-by` trailer that matches the author of the patch. Use Git’s sign-off option:

```bash
git commit -s -m "Your descriptive commit message"
```

That appends a line such as:

```text
Signed-off-by: Your Name <your.email@example.com>
```

By signing off, you assert that you agree to the DCO for that commit.

Full DCO text (same as [developercertificate.org](https://developercertificate.org/)):

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

Contributions consisting of commits without a valid sign-off cannot be merged.

## Development workflow

- Install dependencies with **[uv](https://docs.astral.sh/uv/)** per the top-level [`README.md`](README.md).
- Lint and format (Ruff): `make lint` / `make lint-fix` from the repository root.
- Forecasting tests (example): `cd forecasting && uv run pytest sdk/tests`.

Open a pull request against the default branch with a clear description of the change and any relevant issue references.
