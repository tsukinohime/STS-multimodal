"""STS-experiments — zero-shot Semantic Textual Similarity evaluation.

Research premise: when measuring semantic textual similarity between two texts,
can images generated from each text (used as a caption for an image-generation
model) supply extra semantic information that helps the task?

This package is the **text-only baseline** every later multimodal result is
compared against. Pipeline: sentence pair -> frozen pretrained encoder ->
L2-normalised embeddings -> cosine similarity -> Spearman (headline) and
Pearson correlation against human gold scores. Nothing is trained, fine-tuned,
regressed or calibrated, and no decision is made on a test split.

Modules
-------
``config``       one YAML -> typed configuration
``data``         dataset adapters -> one uniform pair schema
``models``       model adapters   -> one uniform encoder interface
``cache``        content-addressed embedding cache (resume across jobs)
``similarity``   paired cosine similarity
``metrics``      Spearman / Pearson / bootstrap CI / macro average
``pipeline``     orchestration
``manifest``     reproducibility record
``report``       metrics CSV/JSON + markdown summary
``diagnostics``  validation checks and the cross-model alignment probe
``cli``          command-line entry point
"""

__version__ = "0.2.0"
