"""Propose a round's candidates by finding where the context repeats.

An agentic coding model's output repeats its input, and a round's verify slots
are nearly free: sparse attention runs one core group a row with a one-row
critical path, so rows past the eighth cost 0.112 ms each while two thirds of
the cube idles. Replayed against a finished run, a lookup of this shape
accepts 5.885 tokens a round at width 32 against the MTP chain's 4.511 at
width 8, and it issues no device work at all.

It proposes a single causal run, which is the same shape the chain proposes,
so the target verifies it unchanged -- no tree mask, no new kernel.

Depth is exhausted at that width: k=63 reaches 5.988. What caps it is
coverage, since 39.7% of rounds find no repeat, so this is a drafter to pair
with a model rather than to run alone.
[evidence](../../../docs/evidence/ascend-decode/lookup-drafting-20261019)
"""


class LookupDrafter:
    """Hold the context and propose what followed its tail last time.

    The context is the prompt plus every committed token, which is what the
    model attends to and therefore what it can copy from.
    """

    def __init__(self, prompt, longest=8, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Tail lengths must satisfy 1 <= shortest <= "
                             "longest")
        self.context = list(prompt)
        self.longest = longest
        self.shortest = shortest
        self.hits = 0
        self.rounds = 0

    def commit(self, tokens):
        """Extend the context with what the target actually accepted."""
        self.context.extend(int(token) for token in tokens)

    def propose(self, count):
        """Return up to `count` tokens, longest tail match first.

        A longer repeated tail predicts its continuation better, so the
        search runs from `longest` down and takes the most recent occurrence
        at the first length that has one.
        """
        self.rounds += 1
        context = self.context
        for size in range(self.longest, self.shortest - 1, -1):
            if len(context) <= size:
                continue
            tail = context[-size:]
            for start in range(len(context) - size - 1, -1, -1):
                if context[start:start + size] != tail:
                    continue
                run = context[start + size:start + size + count]
                if run:
                    self.hits += 1
                    return run
        return []

    def report(self):
        return {
            "drafter": "lookup",
            "longest_tail": self.longest,
            "shortest_tail": self.shortest,
            "context_tokens": len(self.context),
            "rounds": self.rounds,
            "tail_hits": self.hits,
        }
