"""STS-experiments: zero-shot Semantic Textual Similarity evaluation framework.

Milestone 1 (text-only reference): encode two texts with a pre-trained text
encoder, take the cosine similarity of their embeddings, and correlate those
predictions with human gold STS scores (Spearman / Pearson).

Later milestones add a generated-image (visual) modality on top of this same
pipeline; see the project research premise in memory.
"""

__version__ = "0.1.0"
