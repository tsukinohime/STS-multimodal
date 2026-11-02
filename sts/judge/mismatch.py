"""Condition E2: which other pair's images each sentence pair is shown.

E2 keeps D2's judge and prompt verbatim (the sentence pair, then two generated
images introduced as the images of those sentences) but fills the two image
slots with the images generated for *another* pair of the same dataset, its
donor. The donors are a random derangement of the dataset's pairs, drawn from
``mismatch_seed`` and the dataset name:

* Every pair's two images are shown exactly once, so E2 shows D2's image pairs
  and changes only which sentence pair each one accompanies.
* No pair is shown its own images, nor those of a pair that shares a sentence
  with it (exact text) or, in CxC, a COCO image: those would be partly its own.
* The donor keeps its slot order: its first sentence's image fills slot one.
* The same seed and pairs give the same donors on any machine, and every
  prediction row names its donor (``donor_pair_id``) and the donor's gold score.

Each dataset is deranged on its own, so ``sick-test`` and ``sick-all`` give
their shared pairs different donors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

from ..data import STSDataset, STSPair
from ..hashing import sha256_text
from ..metrics import compute_correlations

#: Repair passes before giving up on a dataset whose pairs leave no valid donor.
_MAX_ROUNDS = 100
#: Random partners tried per invalid pair and pass.
_CANDIDATES = 64


@dataclass(frozen=True)
class DonorAssignment:
    """For each pair of ``dataset``, the pair whose two generated images it is shown."""

    dataset: STSDataset
    donor: np.ndarray          # donor[i]: index of the pair whose images pair i is shown
    seed: int

    def pair_of(self, index: int) -> STSPair:
        """The donor of pair ``index``."""
        return self.dataset.pairs[int(self.donor[index])]

    @property
    def images(self) -> Tuple[List[str], List[str]]:
        """The sentences whose generated images fill each pair's two slots, aligned with the pairs."""
        donors = [self.dataset.pairs[j] for j in self.donor]
        return [p.sentence1 for p in donors], [p.sentence2 for p in donors]

    def columns(self) -> Dict[str, object]:
        """Prediction-file columns recording which images every pair was shown."""
        donors = [self.dataset.pairs[j] for j in self.donor]
        return {
            "images_from": "other-pair",
            "mismatch_seed": self.seed,
            "donor_pair_id": [p.pair_id for p in donors],
            "donor_sentence1_id": [p.sentence1_id for p in donors],
            "donor_sentence2_id": [p.sentence2_id for p in donors],
            "donor_gold_score": [p.gold_score for p in donors],
        }

    def diagnostics(self, expected: np.ndarray) -> Dict[str, object]:
        """Spearman of the donor's gold with the pair's gold (the randomisation, ~0) and with the score.

        Donor gold is independent of the pair's own gold, so a score that
        correlates with it follows the images.
        """
        gold = self.dataset.gold
        donor_gold = gold[self.donor]
        finite = np.isfinite(expected)
        return {
            "seed": self.seed,
            "n_pairs": int(self.donor.size),
            "spearman_gold_vs_donor_gold": compute_correlations(gold, donor_gold).spearman,
            "spearman_expected_vs_donor_gold": compute_correlations(expected[finite], donor_gold[finite]).spearman,
        }


def _identity(pair: STSPair) -> FrozenSet[Tuple[str, str]]:
    """What a pair's images could share with another pair's: its sentences and, in CxC, its COCO images."""
    keys = {("text", pair.sentence1), ("text", pair.sentence2)}
    keys.update(("coco", image) for image in (pair.image1_id, pair.image2_id) if image)
    return frozenset(keys)


def assign_donors(dataset: STSDataset, seed: int) -> DonorAssignment:
    """A seeded random derangement of ``dataset``'s pairs in which no pair is related to its donor.

    A uniform random permutation first; then each pair left with its own or a
    related pair's images swaps donors with a random partner, whenever the swap
    leaves both pairs valid. Swaps keep the permutation a permutation.
    """
    n = len(dataset)
    if n < 2:
        raise ValueError(f"{dataset.name}: {n} pair(s); mismatched images need at least two")
    identities = [_identity(pair) for pair in dataset.pairs]
    rng = np.random.default_rng([seed, int(sha256_text(dataset.name)[:8], 16)])
    donor = rng.permutation(n)

    def related(i: int, j: int) -> bool:
        return i == j or not identities[i].isdisjoint(identities[j])

    for _ in range(_MAX_ROUNDS):
        invalid = [i for i in range(n) if related(i, donor[i])]
        if not invalid:
            return DonorAssignment(dataset=dataset, donor=donor, seed=seed)
        for i in invalid:
            if not related(i, donor[i]):
                continue  # fixed by an earlier swap in this pass
            for j in rng.integers(0, n, size=_CANDIDATES):
                if not related(i, donor[j]) and not related(j, donor[i]):
                    donor[i], donor[j] = donor[j], donor[i]
                    break
    raise ValueError(f"{dataset.name}: found no assignment in which every pair is shown an unrelated pair's images")
