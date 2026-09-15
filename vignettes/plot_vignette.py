"""
Vignette
========

This vignette derives BCMP domains from an unintegrated single-cell dataset,
then uses these domains to rank candidate integration methods.
"""

# If needed: python -m pip install matplotlib
import matplotlib.pyplot as plt

# sphinx_gallery_start_ignore
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Setting an item of incompatible dtype is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Argument `use_highly_variable` is deprecated.*",
    category=FutureWarning,
)
logging.getLogger("scib_metrics.metrics._kbet").setLevel(logging.WARNING)
# sphinx_gallery_end_ignore

from bcmp import bcmp
from bcmp.datasets import diabetic_kidney_lite


# %%
# 1. Load the AnnData input
# -------------------------
# The bundled example contains raw counts and batch labels in ``obs["batch"]``.
adata = diabetic_kidney_lite()
print(adata)


# %%
# 2. Run BCMP
# -----------
# Simply run BCMP with the AnnData object and its batch key:
result = bcmp(
    adata,
    batch_key="batch",
    verbose=False,
)
adata_bcmp = result.adata
print(adata_bcmp)


# %%
# The resulting domains can be visualized as follows:
coordinates = adata_bcmp.obsm["X_demo_umap"]

_, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for axis, key, title, palette in zip(
    axes,
    ["batch", "bcmp_domain"],
    ["Batch", "BCMP domain"],
    [plt.get_cmap("Accent"), plt.get_cmap("Paired")],
):
    labels = adata_bcmp.obs[key].astype(str)
    for index, label in enumerate(labels.unique()):
        axis.scatter(
            *coordinates[labels.eq(label).to_numpy()].T,
            s=8,
            color=palette(index),
            label=label,
        )
    axis.set_title(title)
    axis.set_axis_off()
    axis.legend()

plt.tight_layout()
plt.show()


# %%
# 3. Rank candidate integration methods
# --------------------------------------
# The bundled candidate embeddings are stored in ``obsm``. Their keys and
# display names are recorded in ``uns["integration_benchmark"]``.
benchmark = adata_bcmp.uns["integration_benchmark"]
embedding_keys = benchmark["embedding_keys"].tolist()
method_names = benchmark["method_names"]


# %%
# BCMP domains are supplied as the population labels for every candidate
# embedding.
from scib_metrics.benchmark import BatchCorrection, Benchmarker

benchmarker = Benchmarker(
    adata_bcmp,
    batch_key="batch",
    label_key="bcmp_domain",
    embedding_obsm_keys=embedding_keys,
    bio_conservation_metrics=None,
    batch_correction_metrics=BatchCorrection(),
    n_jobs=1,
    progress_bar=False,
)
benchmarker.benchmark()
ranking = (
    benchmarker.get_results(min_max_scale=True)
    .drop(index="Metric Type")["Batch correction"]
    .astype(float)
    .sort_values(ascending=False)
)
ranking.index = [method_names[key] for key in ranking.index]
ranking.index.name = "Method"
print(ranking.round(2).to_string())


# %%
# A bar plot provides another view of the ranking:
_, score_ax = plt.subplots(figsize=(7.2, 5.2))
score_ax.barh(ranking.index, ranking, color="0.5")
score_ax.set(xlabel="scIB batch-correction score", xlim=(0, 1), ylabel="")
score_ax.invert_yaxis()
plt.tight_layout()
plt.show()
