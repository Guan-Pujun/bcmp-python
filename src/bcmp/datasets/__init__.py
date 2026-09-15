"""Small datasets bundled with BCMP for documentation and examples."""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["diabetic_kidney_lite"]

_DIABETIC_KIDNEY_LITE_FILE = "diabetic_kidney_lite.h5ad"


def diabetic_kidney_lite() -> AnnData:
    """Load the bundled 1,500-cell diabetic kidney example dataset.

    The returned object contains raw counts in ``X``, batch labels in ``obs``,
    frozen visualization coordinates, and candidate integration embeddings
    used by the bundled vignette. Their keys and display names are stored
    in ``uns['integration_benchmark']``.
    """
    import anndata as ad

    resource = files(__package__).joinpath("_data", _DIABETIC_KIDNEY_LITE_FILE)
    with as_file(resource) as dataset_path:
        return ad.read_h5ad(dataset_path)
