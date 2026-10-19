"""What a context-lookup drafter would have accepted, from a finished run.

A round's acceptance is the whole denominator of TPOT, and the MTP head is not
the only thing that can fill the verify's slots. A context-lookup drafter takes
the last few tokens emitted, finds where that n-gram occurred earlier, and
proposes whatever followed it. No model, no training, no device work -- a
sliding-window compare over token ids.

Whether it is worth building is decided by arithmetic on a record that already
exists: the accepted prefix is the longest common prefix of the proposal and
what the run actually emitted, so the acceptance of a drafter that never ran is
exactly computable. Width is the cost: proposing k tokens needs k + 1 verify
rows.

Scope: a record carries the generated tokens and not the prompt, so this
searches the generation only. A deployed drafter also searches the 8192-token
prompt, which on an agentic trajectory is where the copyable material is. Every
number here is therefore a lower bound.
"""
import argparse
import json
import pathlib
import statistics


def lookup(context, proposal_length, longest, shortest, branches=1):
    """Propose what followed the context's tail, one branch per repeat.

    Longest match first, which is what makes a hit worth trusting: a 4-gram
    repeat predicts its continuation far better than a 1-gram repeat does.

    A tail that repeats several times offers several continuations, and a
    verify wide enough to hold them can take whichever survives. Standard
    prompt lookup keeps only the most recent and discards the rest, which is
    the whole of why it misses two rounds in three.
    """
    found = []
    for size in range(longest, shortest - 1, -1):
        if len(context) <= size:
            continue
        tail = context[-size:]
        for start in range(len(context) - size - 1, -1, -1):
            if context[start:start + size] != tail:
                continue
            branch = context[start + size:start + size + proposal_length]
            if branch and branch not in found:
                found.append(branch)
                if len(found) == branches:
                    return found
        if found:
            return found
    return found


def matched_prefix(proposal, actual):
    count = 0
    for proposed, emitted in zip(proposal, actual):
        if proposed != emitted:
            break
        count += 1
    return count


def simulate(tokens, proposal_length, longest, shortest, branches=1):
    """Walk the stream as rounds, returning each round's matched draft count."""
    matched, position = [], 1
    while position < len(tokens):
        counts = [matched_prefix(branch, tokens[position:])
                  for branch in lookup(tokens[:position], proposal_length,
                                       longest, shortest, branches)]
        count = max(counts, default=0)
        matched.append(count)
        position += count + 1
    return matched


def report(name, matched, width):
    accepted = sum(matched) + len(matched)
    hits = sum(1 for count in matched if count)
    print(f"  {name:<26} width {width:3d}  rounds {len(matched):4d}  "
          f"acceptance {accepted / len(matched):6.3f}  "
          f"hit {hits / len(matched):5.1%}  "
          f"max {max(matched):3d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=pathlib.Path)
    parser.add_argument("--proposal", type=int, nargs="+",
                        default=[7, 15, 23, 31])
    parser.add_argument("--branches", type=int, nargs="+", default=[1])
    parser.add_argument("--longest", type=int, default=8)
    parser.add_argument("--shortest", type=int, default=2)
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    requests = [request for request in record["requests"]
                if not request.get("warmup")]
    for index, request in enumerate(requests):
        tokens = request["tokens"]
        spans = request["session"]["accepted_spans"]
        drafts = [span["matched_drafts"] for span in spans]
        print(f"request {index}: {len(tokens)} tokens")
        report("MTP chain, measured", drafts, spans[0]["width"])
        for length in args.proposal:
            for branches in args.branches:
                matched = simulate(tokens, length, args.longest,
                                   args.shortest, branches)
                report(f"lookup k={length} b={branches}", matched,
                       length * branches + 1)
        print(f"  {'':<22} lookup lengths {args.shortest}..{args.longest}, "
              f"generation only")


if __name__ == "__main__":
    main()
