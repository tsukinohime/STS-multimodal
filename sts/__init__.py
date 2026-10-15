"""STS-experiments — zero-shot Semantic Textual Similarity evaluation.

Research premise: when measuring semantic textual similarity between two texts,
can images generated from each text (used as a caption for an image-generation
model) supply extra semantic information that helps the task?

The text-only baselines every multimodal result is compared against: sentence
pair -> frozen pretrained encoder -> L2-normalised embeddings -> cosine
similarity (or a frozen LLM judge) -> Spearman (headline) and Pearson
correlation against human gold scores. On top of them, condition D1 scores the
images a text-to-image model generates from the two sentences instead of the
sentences, and condition D2 has the judge rate the sentences together with
those images. Nothing is trained, fine-tuned, regressed or calibrated, and no
decision is made on a test split.

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
``judge``        LLM-as-judge (text) and its vision-language variant (images)
``images``       generated-image store, text-to-image generators, ``imagine`` stage
``cli``          command-line entry point
"""

__version__ = "0.2.0"
