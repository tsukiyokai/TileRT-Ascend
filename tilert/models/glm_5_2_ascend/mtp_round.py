"""Greedy MTP rounds over the main-model and MTP CANN interfaces."""

from dataclasses import dataclass

from .mtp_state import MtpState


def token_ids(result):
    values = result.token_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class DraftSeed:
    token: int
    recycle_hidden: object
    indices: object


@dataclass(frozen=True)
class ExecutedRound:
    candidates: tuple[int, ...]
    target_predictions: tuple[int, ...]
    commit: object


class MtpRoundExecutor:
    """Run fixed-width greedy rounds without importing a device library.

    ``target.verify`` and ``draft.forward`` return greedy ``token_ids``.
    Main ``last_hidden`` and MTP ``recycle_hidden`` are normalized states.
    All sixteen ranks call this loop in the same order. A failed device
    round requires fresh prefix injection before starting another request.
    """

    def __init__(
        self, target, draft, width=4, max_new_tokens=1024, eos_tokens=(),
    ):
        if width not in (2, 4):
            raise ValueError("MTP rounds require width 2 or 4")
        if target.capacity != draft.capacity:
            raise ValueError("Main and MTP cache capacities must match")
        self.target = target
        self.draft = draft
        self.state = MtpState(
            width, target.capacity, max_new_tokens, eos_tokens
        )
        self.seed = None
        self.seed_parent_hidden = None
        self.failed = False

    def start(self, first_token, normalized_parent_hidden):
        """Start at main length P with MTP length P-1 or P.

        The first token was produced by prefill. Its normalized parent
        hidden is from the last prompt position, P-1. Recomputing that MTP
        row supplies the first draft token, recycle state and index set.
        """
        self.state.reset(self.target.position, first_token,
                         draft_prefix_length=self.draft.position)
        self.seed_parent_hidden = normalized_parent_hidden
        self.seed = None
        self.failed = False

    @staticmethod
    def _seed(result, row):
        return DraftSeed(
            token_ids(result)[row], result.recycle_hidden[row:row + 1],
            result.indices[row:row + 1],
        )

    def advance(self):
        """Propose, verify, recompute MTP parents, and commit one prefix.

        Accepted tokens include a terminating EOS when present; serving
        filters that ID before delivering text. Chained draft KV is never
        committed as target-conditioned KV, even for an accepted token.
        """
        if self.failed:
            raise RuntimeError("Inject fresh prefixes after a failed round")
        if (self.target.position != self.state.main_length
                or self.draft.position != self.state.draft_length):
            raise ValueError("Engine lengths differ from the MTP state")
        plan = self.state.begin()
        try:
            if self.seed is None:
                if self.seed_parent_hidden is None:
                    raise ValueError("MTP seed needs the parent hidden")
                self.draft.truncate(plan.main_start - 1)
                seeded = self.draft.forward(
                    (plan.pending_token,), self.seed_parent_hidden,
                    full_logits=False,
                )
                self.seed = self._seed(seeded, 0)
            seed = self.seed
            candidates = [plan.pending_token]
            shared_indices = seed.indices
            while len(candidates) < plan.width:
                candidates.append(seed.token)
                if len(candidates) < plan.width:
                    chained = self.draft.forward(
                        (seed.token,), seed.recycle_hidden,
                        full_logits=False, external_indices=shared_indices,
                    )
                    seed = self._seed(chained, 0)
            verified = self.target.verify(candidates, full_logits=False)
            predictions = token_ids(verified)
            self.draft.truncate(plan.main_start)
            conditioned = self.draft.forward(
                predictions, verified.last_hidden, full_logits=False,
            )
            commit = self.state.resolve(plan, candidates, predictions)
            self.target.truncate(commit.main.retained)
            self.draft.truncate(commit.draft.retained)
            self.seed = self._seed(conditioned, commit.committed - 1)
            self.seed_parent_hidden = None
            return ExecutedRound(tuple(candidates), predictions, commit)
        except Exception:
            self.failed = True
            raise
