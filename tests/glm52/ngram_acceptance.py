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


def lookup(context, proposal_length, longest, shortest):
    """Propose what followed the most recent repeat of the context's tail.

    Longest match first, which is what makes a hit worth trusting: a 4-gram
    repeat predicts its continuation far better than a 1-gram repeat does.
    """
    for size in range(longest, shortest - 1, -1):
        if len(context) <= size:
            continue
        tail = context[-size:]
        for start in range(len(context) - size - 1, -1, -1):
            if context[start:start + size] == tail:
                return context[start + size:start + size + proposal_length]
    return []


def matched_prefix(proposal, actual):
    count = 0
    for proposed, emitted in zip(proposal, actual):
        if proposed != emitted:
            break
        count += 1
    return count


def simulate(tokens, proposal_length, longest, shortest):
    """Walk the stream as rounds, returning each round's matched draft count."""
    matched, position = [], 1
    while position < len(tokens):
        proposal = lookup(tokens[:position], proposal_length,
                          longest, shortest)
        count = matched_prefix(proposal, tokens[position:])
        matched.append(count)
        position += count + 1
    return matched


def report(name, matched, width):
    accepted = sum(matched) + len(matched)
    hits = sum(1 for count in matched if count)
    print(f"  {name:<22} width {width:3d}  rounds {len(matched):4d}  "
          f"acceptance {accepted / len(matched):6.3f}  "
          f"hit {hits / len(matched):5.1%}  "
          f"max {max(matched):3d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=pathlib.Path)
    parser.add_argument("--proposal", type=int, nargs="+",
                        default=[7, 15, 23, 31])
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
            matched = simulate(tokens, length, args.longest, args.shortest)
            report(f"context lookup k={length}", matched, length + 1)
        print(f"  {'':<22} lookup lengths {args.shortest}..{args.longest}, "
              f"generation only")


if __name__ == "__main__":
    main()
