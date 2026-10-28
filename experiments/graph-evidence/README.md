# CPU graph evidence staging

`frozen/` preserves the executed sources byte for byte. The runner relocates
only path literals, saves configured sources beside a new output directory,
and uses the archived msprobe wrapper/schema sources. It requires the tested
CPU environment: PyTorch 2.9, Transformers 5.12, and safetensors.

```sh
python experiments/graph-evidence/run_frame32.py --output /tmp/frame32-new --check-only
python experiments/graph-evidence/run_frame32.py --output /tmp/frame32-new
python experiments/graph-evidence/audit.py
python experiments/graph-evidence/check_readback.py --output /tmp/frame32-readback.json
```

The [saved experiment](../../docs/evidence/ascend-decode/graph-frame32-20261028/README.md)
owns the field inventory, results, and remaining producer checks. The audit
reads every compressed byte and all recorded checks without running HF.

`receiver.decode_frame()` is the current reusable receiver. It requires an
independently admitted reference with exactly 32 rows, checks typed identities,
and returns owned CPU snapshots. `check_readback.py` reads the saved real
tensors, recomputes the scalar oracles, and checks rejection cases without
constructing or running HF. Formal CANN admission remains a separate owner.
