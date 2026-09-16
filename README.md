<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/bcmp-github-dark.svg">
    <img src="docs/_static/bcmp-github-light.svg" alt="BCMP — Batch-Constraint Manifold Partition" width="460">
  </picture>
</p>

<br>

<p align="center">
  <a href="https://github.com/Guan-Pujun/bcmp-python/actions/workflows/ci.yml"><img src="https://github.com/Guan-Pujun/bcmp-python/actions/workflows/ci.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://github.com/Guan-Pujun/bcmp-python/actions/workflows/package.yml"><img src="https://github.com/Guan-Pujun/bcmp-python/actions/workflows/package.yml/badge.svg?branch=main" alt="Build"></a>
  <a href="https://github.com/Guan-Pujun/bcmp-python/actions/workflows/docs.yml"><img src="https://github.com/Guan-Pujun/bcmp-python/actions/workflows/docs.yml/badge.svg?branch=main" alt="Documentation"></a>
  <br>
  <a href="https://codecov.io/github/Guan-Pujun/bcmp-python"><img src="https://codecov.io/github/Guan-Pujun/bcmp-python/graph/badge.svg?token=9MGXPGWRVI&amp;color=brightgreen" alt="Codecov"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=Guan-Pujun_bcmp-python"><img src="https://sonarcloud.io/api/project_badges/measure?project=Guan-Pujun_bcmp-python&amp;metric=alert_status&amp;token=80ac6bae4b632a9fc13f1b7be053d2212421e76e" alt="SonarQube Quality Gate"></a>
  <a href="https://www.bestpractices.dev/projects/14665"><img src="https://www.bestpractices.dev/projects/14665/badge" alt="OpenSSF Best Practices"></a>
</p>

<br>

# BCMP<sup>Py</sup>

The Python implementation of Batch-Constraint Manifold Partition (BCMP).
BCMP derives proxy population labels from unintegrated single-cell data
geometry for evaluation of batch-effect removal across candidate integration
methods.

## Installation

```bash
python -m pip install bcmp
```

For macOS on Apple Silicon, follow the
[installation guide](https://guan-pujun.github.io/bcmp-python/installation.html).

## Quick start

Provide an `AnnData` object with raw counts in `adata.X` and batch labels in
`adata.obs["batch"]`:

```python
from bcmp import bcmp

result = bcmp(adata, batch_key="batch")
domains = result.adata.obs["bcmp_domain"]
```

## Manual

The [BCMP<sup>Py</sup> manual](https://guan-pujun.github.io/bcmp-python/)
describes installation, the public API, returned results, and the vignette.

## Feedback and contributions

Report bugs or request enhancements through the
[GitHub issue tracker](https://github.com/Guan-Pujun/bcmp-python/issues).
For proposed changes, see [CONTRIBUTING.md](CONTRIBUTING.md). Please report
suspected security vulnerabilities through the private process in
[SECURITY.md](SECURITY.md), not in a public issue.

## Citation

Citation information will be added when available.

## License

BCMP is distributed under the [GPL-3.0-or-later license](LICENSE).
