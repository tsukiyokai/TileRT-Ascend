"""Check profiling lifetime and lookup accounting without NPU devices."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_cann_mtp as bench  # noqa: E402
import read_levers  # noqa: E402
from check_mtp_tp import make_backend, serial_oracle  # noqa: E402
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402


class BenchmarkChecks(unittest.TestCase):
    def drive(self, output, fail_during_capture=False):
        backend, ranks, channel = make_backend("mixed", capacity=32)
        prompt = (3, 10, 17)
        first = serial_oracle(prompt, 1, 32)[0][0]
        args = SimpleNamespace(width=4, warmup=1, requests=2,
                               max_tokens=9, rounds_per_command=1,
                               profile_last=True, output=output)
        events = []
        active = [False]

        class Capture:
            def __enter__(capture):
                events.append(("enter", channel.commands[-1]["operation"],
                               backend.position))
                active[0] = True
                return capture

            def __exit__(capture, error_type, error, traceback):
                events.append(("exit", error_type))
                active[0] = False

        profiler = SimpleNamespace(
            profile=lambda **kwargs: Capture(),
            ProfilerActivity=SimpleNamespace(NPU="npu"),
            ProfilerLevel=SimpleNamespace(Level1=1),
            _ExperimentalConfig=lambda **kwargs: kwargs,
            tensorboard_trace_handler=lambda path: path,
            profiler=SimpleNamespace(analyse=lambda **kwargs:
                                     events.append(("analyse", kwargs))),
        )
        advance = backend.advance_mtp_n

        def advance_or_fail(count):
            if fail_during_capture and active[0]:
                raise RuntimeError("decode failed during capture")
            return advance(count)

        def request(rid, length):
            return PrefixRequest(rid, length, {"prompt": prompt})

        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(profiler=profiler)}), \
                patch.object(bench, "PREFIX_LENGTH", len(prompt)), \
                patch.object(bench, "PrefixRequest", side_effect=request), \
                patch.object(backend, "advance_mtp_n", side_effect=advance_or_fail):
            if fail_during_capture:
                with self.assertRaisesRegex(RuntimeError, "decode failed during capture"):
                    bench.drive(backend, first, args)
                records = None
            else:
                records = bench.drive(backend, first, args)
        self.assertTrue(all(rank.closed for rank in ranks))
        self.assertTrue(all(rank.engine.position == 0 for rank in ranks))
        self.assertFalse(active[0])
        return records, events, args

    def test_real_session_profiles_only_last_decode_and_excludes_its_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            records, events, args = self.drive(Path(directory))
        self.assertEqual([row["warmup"] for row in records], [True, False, False])
        self.assertEqual([row["profiled"] for row in records], [False, False, True])
        self.assertEqual(events, [
            ("enter", "inject", 3), ("exit", None),
            ("analyse", {"profiler_path": str(args.output / "capture")}),
        ])
        self.assertEqual(bench.timing_requests(records), [records[1]])
        self.assertTrue(all(row["tokens"] == row["returned_tokens"] for row in records))
        self.assertTrue(all(row["operations"][-1]["operation"] == "reset"
                            for row in records))

    def test_decode_failure_closes_profiler_and_resets_the_session(self):
        with tempfile.TemporaryDirectory() as directory:
            _, events, _ = self.drive(Path(directory), fail_during_capture=True)
        self.assertEqual(events, [("enter", "inject", 3), ("exit", RuntimeError)])

    def state(self, lookup):
        draft = None if lookup else 8193
        reset = [0, None if lookup else 0]
        request = {
            "first_token": 7, "injected": [8192, None if lookup else 8191],
            "finished": [8193, draft], "reset": reset,
            "rounds": [{
                "width": 2, "candidates": [7, 99], "predictions": [8, 100],
                "accepted_tokens": [8], "matched_drafts": 0,
                "natural_accepted": 1, "before": 8192, "written": 8194,
                "retained": 8193, "draft_retained": draft,
                "start_ns": 1, "end_ns": 2, "elapsed_ns": 1,
            }],
        }
        args = SimpleNamespace(lookup=lookup, warmup=0, requests=1,
                               max_tokens=2, width=2, target_graph=False,
                               draft_graph=False, conditioned_graph=False,
                               automaton=False, tree=False)
        report = {"rank": 1, "closed": True, "final_positions": reset,
                  "rank_requests": [request], "command_serial": 6}
        return args, report

    def test_lookup_accepts_absent_draft_and_rejects_fabricated_draft_state(self):
        args, report = self.state(lookup=True)
        self.assertIsNone(bench.MeasuredRank.draft_position.fget(
            SimpleNamespace(draft=None)))
        self.assertEqual(bench.validate_local(report, args)[0]["tokens"], [7, 8])
        for key, value in (("injected", [8192, 8191]), ("reset", [0, 0]),
                           ("finished", [8193, 8193])):
            corrupted = deepcopy(report)
            corrupted["rank_requests"][0][key] = value
            with self.assertRaises(ValueError):
                bench.validate_local(corrupted, args)
        corrupted = deepcopy(report)
        corrupted["final_positions"] = [0, 0]
        with self.assertRaises(ValueError):
            bench.validate_local(corrupted, args)

    def test_mtp_still_requires_the_draft_positions(self):
        args, report = self.state(lookup=False)
        self.assertEqual(bench.validate_local(report, args)[0]["tokens"], [7, 8])
        report["rank_requests"][0]["injected"][1] = None
        with self.assertRaises(ValueError):
            bench.validate_local(report, args)

    def test_only_device_drafting_loads_the_mtp_model(self):
        target = object()
        modes = ((True, False, False), (False, True, False),
                 (False, False, True), (False, False, False))
        for lookup, automaton, tree in modes:
            args = SimpleNamespace(lookup=lookup, automaton=automaton,
                                   tree=tree, draft_projection_tp=True)
            candidate = SimpleNamespace(
                model=SimpleNamespace(do_not_compile=True),
            )
            with self.subTest(mode=(lookup, automaton, tree)), patch(
                "tilert.models.glm_5_2_ascend.cann_mtp.CannMtp",
                return_value=candidate,
            ) as load:
                draft = bench.create_draft(target, args)
                if lookup or automaton or tree:
                    self.assertIsNone(draft)
                    load.assert_not_called()
                else:
                    self.assertIs(draft, candidate)
                    load.assert_called_once_with(target,
                                                 shard_projections=True)

    def test_host_modes_reject_unused_mtp_flags_before_loading(self):
        required = ["bench", "--checkpoint", "checkpoint", "--capture-case",
                    "prefix", "--output", "out", "--source-commit", "test"]
        for mode in ("--lookup", "--automaton", "--tree"):
            for options in (("--draft-graph",), ("--draft-projection-tp",),
                            ("--mtp-tokens", "1")):
                with (
                    self.subTest(mode=mode, options=options),
                    patch.object(sys, "argv", [*required, mode, *options]),
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        bench.arguments()

    def test_cli_requires_a_sample_without_profiler(self):
        command = ["bench", "--checkpoint", "checkpoint", "--capture-case", "prefix",
                   "--output", "out", "--source-commit", "test", "--requests", "1"]
        with patch.object(sys, "argv", command):
            self.assertEqual(bench.arguments().requests, 1)
        with patch.object(sys, "argv", command + ["--profile-last"]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            bench.arguments()
        self.assertEqual(error.exception.code, 2)

    def test_saved_result_readers_exclude_profiled_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(1, 4):
                for name, elapsed in (("control", 10), ("candidate", 12)):
                    output = root / f"{name}-{number}"
                    output.mkdir()
                    records = [{
                        "warmup": False, "tpot_ms": 2, "output_tokens": 9,
                        "operations": [{"operation": "advance_mtp_n",
                                        "elapsed_ns": elapsed * 1000000}],
                    }]
                    profiled = deepcopy(records[0])
                    profiled.update(profiled=True, tpot_ms=1000)
                    profiled["operations"][0]["elapsed_ns"] = 1
                    records.append(profiled)
                    (output / "rank-0.json").write_text(json.dumps({
                        "requests": records}))
                    result = read_levers.read(output)
                    self.assertEqual(result["tpot"], 2)
                    self.assertEqual(result["round"], elapsed)
                    self.assertEqual(result["rounds"], 1)
            result = subprocess.check_output([
                sys.executable, str(Path(__file__).with_name("paired.py")),
                str(root), "control", "candidate"], text=True)
            self.assertIn("paired mean delta +2.000 ms a round", result)


if __name__ == "__main__":
    unittest.main()
