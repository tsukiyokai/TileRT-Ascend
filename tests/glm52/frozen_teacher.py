"""Fixed-input target diagnostics, independent of free-running acceptance."""

import hashlib
import json


def load_teacher(path, capture, width, capacity):
    teacher = json.loads(path.read_text())
    manifest = json.loads((capture / "manifest.json").read_text())
    tokens = teacher["teacher_tokens"]
    expected = teacher["expected_next_tokens"]
    # The baseline hashes its JSON prompt file, including the final newline.
    prompt = (json.dumps(manifest["prompt_token_ids"]) + "\n").encode()
    if (teacher["prompt_sha256"] != hashlib.sha256(prompt).hexdigest()
            or teacher["prompt_tokens"] != manifest["seq_len"]
            or teacher["first_token_id"] != manifest["first_token_id"]):
        raise ValueError("Frozen teacher and capture describe different inputs")
    if (len(tokens) != 32 or len(expected) != 32
            or len(tokens) % width
            or tokens[0] != teacher["first_token_id"]
            or tokens[1:] != expected[:-1]
            or any(type(token) is not int or not 0 <= token < 154880
                   for token in tokens + expected)):
        raise ValueError("All 32 teacher inputs and successors are required")
    if manifest["seq_len"] + len(tokens) > capacity:
        raise ValueError("Teacher inputs exceed cache capacity")
    return {**teacher, "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def teacher_run(target, teacher, width, directory, observe):
    import torch

    tokens = teacher["teacher_tokens"]
    expected = teacher["expected_next_tokens"]
    initial = target.position
    rows, first_difference, first_artifact = [], None, None
    for begin in range(0, len(tokens), width):
        ids = tuple(tokens[begin:begin + width])
        result = target.verify(ids, full_logits=True)
        predictions = result.token_ids.tolist()
        observe((begin, ids, predictions, target.position))
        logits = result.logits.detach().float().cpu()
        if (len(predictions) != width or logits.ndim != 2
                or logits.shape[0] != width
                or result.last_hidden.shape[0] != width):
            raise ValueError("Teacher output rows differ from consumed inputs")
        if not torch.isfinite(logits).all():
            torch.save(logits, directory / f"teacher-m{width}-nonfinite.pt")
            raise ValueError("Nonfinite teacher logits")
        if predictions != logits.argmax(dim=-1).tolist():
            raise ValueError("Teacher predictions differ from logits argmax")
        top_values, top_ids = logits.topk(2, dim=-1)
        for row, prediction in enumerate(predictions):
            step = begin + row
            label = expected[step]
            rows.append({
                "step": step, "position": initial + step,
                "input": tokens[step], "expected": label,
                "prediction": prediction,
                "top2_ids": top_ids[row].tolist(),
                "top2_logits": top_values[row].tolist(),
                "expected_logit": float(logits[row, label]),
                "top1_margin": float(top_values[row, 0] - top_values[row, 1]),
                "expected_gap": float(logits[row, prediction]
                                      - logits[row, label]),
            })
            if prediction != label and first_difference is None:
                first_difference = step
                hidden = result.last_hidden[row].detach().cpu().clone()
                path = directory / f"teacher-m{width}-first-difference.pt"
                torch.save({
                    **rows[-1], "logits": logits[row].clone(),
                    "last_hidden": hidden,
                }, path)
                first_artifact = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    return {
        "width": width, "rows": rows, "first_difference": first_difference,
        "first_difference_file": first_artifact,
        "initial_position": initial, "final_position": target.position,
        "passed": first_difference is None and len(rows) == len(tokens)
        and target.position == initial + len(tokens),
    }
