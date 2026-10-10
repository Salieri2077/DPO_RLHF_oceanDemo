"""M1b: OceanDepths tools, handle registry, scorer and SDK integration.

Most cases run on a small in-memory dataset with hand-made profiles, so the support situations the dev set does not
contain (profiles but zero estimable, profiles but zero pairs) are covered without changing the dev requests.
A few cases replay real dev requests when the local OceanDepths copy is present.
"""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HAS_SDK = importlib.util.find_spec("agents") is not None
ROOT = Path(__file__).resolve().parents[1]
DEPTH = np.array([0.494025, 1.541375, 2.645669, 3.819495, 5.078224, 6.440614, 7.92956, 9.572997, 11.405, 13.46714,
                  15.81007, 18.49556, 21.59882, 25.21141, 29.44473, 34.43415, 40.34405, 47.37369, 55.76429, 65.80727,
                  77.85385, 92.32607, 109.7293, 130.666, 155.8507, 186.1256, 222.4752, 266.0403, 318.1274, 380.213,
                  453.9377, 541.0889, 643.5668, 763.3331, 902.3393, 1062.44, 1245.291, 1452.251, 1684.284, 1941.893,
                  2225.078, 2533.336, 2865.703, 3220.82, 3597.032, 3992.484, 4405.224, 4833.291, 5274.784, 5727.917])
REGION = [[10.0, 20.0], [150.0, 160.0]]

if HAS_SDK:
    from transformers import AutoTokenizer
    from agent.oceanlab import env as envmod
    from agent.oceanlab.data import OceanDepths, query_partition, split_dates
    from agent.oceanlab.env import OceanLab, OceanLabHarness, observe
    from agent.oceanlab.oracle import oracle_texts, tool_text
    from agent.oceanlab.scoring import Scorer, aggregate
    from agent.oceanlab.tasks import reference_result, request_status
    from agent.ocean import read_jsonl
    from agent.sdk import run_trajectory

    class FakeData(OceanDepths):
        """In-memory stand-in with the store's array names; everything above `read` is the real code."""
        def __init__(self, profiles):
            n = len(profiles)
            juld = [(pd.Timestamp(p["date"]) + pd.Timedelta(hours=12) - pd.Timestamp("1950-01-01")) / pd.Timedelta(days=1)
                    for p in profiles]
            frame = pd.DataFrame({"profile": np.arange(n), "latitude": [p["lat"] for p in profiles],
                                  "longitude": [p["lon"] for p in profiles], "profile_juld": juld,
                                  "glorys_temporal_status": [p.get("status", 0) for p in profiles],
                                  "profile_source_file": ["fake.nc"] * n, "profile_idx": np.arange(n)})
            frame["time"] = pd.Timestamp("1950-01-01") + pd.to_timedelta(frame["profile_juld"], unit="D")
            self._index, self.arrays = frame, {}
            theta = np.stack([20 - 0.05 * DEPTH for _ in profiles]) if n else np.zeros((0, 50))
            salt = np.full((n, 50), 35.0)
            ok = np.ones((n, 50), dtype=bool)
            for i, p in enumerate(profiles):
                for k in p.get("missing_levels", []):
                    ok[i, k] = False
            for prefix, value in (("potm", theta), ("psal", salt), ("temp", theta + 0.01)):
                self.arrays[f"argo_{prefix}_on_glorys_depth"] = np.where(ok, value, np.nan).astype(np.float32)
                self.arrays[f"argo_{prefix}_valid_on_glorys_depth"] = ok.astype(np.int8)
                self.arrays[f"argo_{prefix}_qc_on_glorys_depth"] = np.where(ok, 1, -1).astype(np.int8)
            self.arrays["argo_profile_potm_qc"] = np.ones(n, np.int8)
            self.arrays["argo_profile_psal_qc"] = np.ones(n, np.int8)
            self.arrays["argo_position_qc"] = np.array([p.get("position_qc", 1) for p in profiles], np.int8)
            self.arrays["argo_juld_qc"] = np.full(n, -1, np.int8)
            self.arrays["argo_profile_depth_qc"] = np.full(n, -1, np.int8)
            self.arrays["argo_depth_qc_on_glorys_depth"] = np.full((n, 50), -1, np.int8)
            self.arrays["glorys_thetao"] = (theta + 0.1).astype(np.float32)
            self.arrays["glorys_so"] = (salt - 0.02).astype(np.float32)

        index = property(lambda self: self._index)
        depth = property(lambda self: DEPTH)

        def read(self, name, rows):
            return self.arrays[name][np.asarray(rows, dtype=int)]

    def fake_profiles():
        good = dict(lat=12.0, lon=152.0)
        return [
            {**good, "date": "2018-03-05"}, {**good, "date": "2018-03-17"}, {**good, "date": "2018-03-29"},
            {**good, "date": "2018-03-30", "position_qc": 4},           # excluded by qc_policy_v1 only
            # April: profiles exist, but no 10 m reference (9.573 m missing) and boundary-only collocation
            {**good, "date": "2018-04-10", "missing_levels": [7], "status": 1},
            {**good, "date": "2018-04-20", "missing_levels": [7], "status": 1},
            # May: none
        ]

    def spec_for(data, groups, analyses, consistency=None, sid="fake-000"):
        spec = {"id": sid, "family": "report" if len(analyses) == 2 else
                ("mld" if analyses == ["mld"] else "consistency_temperature"),
                "region": REGION, "groups": [{"label": m[0] if len(m) == 1 else f"{m[0]}..{m[-1]}", "months": m}
                                             for m in groups],
                "analyses": analyses, "text": "fake request"}
        if consistency:
            spec["consistency"] = consistency
        spec["reference"], agree = reference_result(data, spec)
        assert agree
        return spec


@unittest.skipUnless(HAS_SDK, "optional SDK environment required")
class FakeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(ROOT / "model")
        cls.data = FakeData(fake_profiles())
        cls.lab = OceanLab("dev", cls.data)
        cls.harness = OceanLabHarness(Scorer(cls.data))

    def generator(self, texts):
        sequence = iter(texts)

        def generate(ids, limit):
            tokens = self.tok.encode(next(sequence), add_special_tokens=False) + [self.tok.eos_token_id]
            return tokens, []
        return generate

    def episode(self, spec, texts):
        return run_trajectory(spec, self.tok, self.lab, self.generator(texts), harness=self.harness)

    def query(self, start, end):
        (a, b), (c, d) = REGION
        return tool_text("查询", "query_profiles", {"lat_min": a, "lat_max": b, "lon_min": c, "lon_max": d,
                                                    "start_date": start, "end_date": end})

    def mld(self, groups, qc="qc_policy_v1", dataset="ds_1"):
        return tool_text("计算", "compute_mld", {"dataset": dataset, "variable": "potential_temperature",
                                                 "qc_policy": qc, "groups": groups})

    @staticmethod
    def final(status, **results):
        return "<final>" + json.dumps({"status": status, "results": results}) + "</final>"

    def test_prompt_uses_this_environments_system_and_tools(self):
        spec = spec_for(self.data, [["2018-03"]], ["mld"])
        trace = self.episode(spec, oracle_texts(spec, self.lab))
        for entry in trace["rounds"]:
            prompt = self.tok.decode(entry["input_ids"])
            self.assertIn(envmod.SYSTEM, prompt)
            self.assertIn(spec["text"], prompt)
            for tool in envmod.TOOLS:
                self.assertIn(json.dumps(tool["function"]["parameters"], ensure_ascii=False)[:60], prompt)
            for old in ("marine_calculate", "search_ocean", "OceanHeart海洋任务助手"):
                self.assertNotIn(old, prompt)

    def test_zero_estimable_with_profiles_is_insufficient(self):
        spec = spec_for(self.data, [["2018-04"]], ["mld"])
        self.assertEqual(spec["reference"]["mld"]["2018-04"]["profiles"], 2)
        self.assertEqual(spec["reference"]["status"], "insufficient")
        trace = self.episode(spec, oracle_texts(spec, self.lab))
        self.assertEqual(trace["metrics"]["success"], 1, trace["metrics"])
        self.assertEqual(json.loads(trace["final"])["status"], "insufficient")
        april = [{"label": "4月", "start_date": "2018-04-01", "end_date": "2018-04-30"}]
        cases = {"no evidence": ([self.final("insufficient")], "missing_analysis"),
                 "wrong status": ([self.query("2018-04-01", "2018-04-30"), self.mld(april),
                                   self.final("complete", mld="res_1")], "status"),
                 "made-up handle": ([self.final("insufficient", mld="res_1")], "unknown_handle")}
        for name, (texts, failure) in cases.items():
            with self.subTest(name):
                metrics = self.episode(spec, texts)["metrics"]
                self.assertEqual((metrics["success"], metrics["failure"]), (0, failure))

    def test_zero_pairs_with_profiles_is_insufficient(self):
        spec = spec_for(self.data, [["2018-04"]], ["consistency"],
                        {"variable": "potential_temperature", "shallow_requested": False, "depth_range_m": [5.0, 50.0]})
        self.assertEqual(spec["reference"]["status"], "insufficient")
        self.assertEqual(spec["reference"]["consistency"]["2018-04"]["excluded_temporal_status"], 2)
        self.assertEqual(self.episode(spec, oracle_texts(spec, self.lab))["metrics"]["success"], 1)
        groups = [{"label": "a", "start_date": "2018-04-01", "end_date": "2018-04-30"}]
        any_filter = tool_text("诊断", "compute_consistency", {
            "dataset": "ds_1", "variable": "potential_temperature", "qc_policy": "qc_policy_v1",
            "temporal_filter": "any", "depth_min": 5, "depth_max": 50, "groups": groups})
        trace = self.episode(spec, [self.query("2018-04-01", "2018-04-30"), any_filter,
                                    self.final("partial", consistency="res_1")])
        self.assertEqual(trace["calls"][1]["result"]["method_card"], None)
        self.assertGreater(trace["calls"][1]["result"]["groups"][0]["levels_with_pairs"], 0)
        self.assertEqual(trace["metrics"]["parameter_errors"], {"consistency": ["temporal_filter"]})
        self.assertEqual(trace["metrics"]["env_error"], 0)

    def test_consistency_handle_records_exclusion_reasons(self):
        episode = self.lab.episode()
        ds = episode.execute({"name": "query_profiles", "arguments": {
            "lat_min": 10, "lat_max": 20, "lon_min": 150, "lon_max": 160,
            "start_date": "2018-03-01", "end_date": "2018-05-31"}})["dataset"]
        groups = [{"label": m, "start_date": f"2018-{m}-01", "end_date": f"2018-{m}-{d}"}
                  for m, d in (("03", 31), ("04", 30), ("05", 31))]
        out = episode.execute({"name": "compute_consistency", "arguments": {
            "dataset": ds, "variable": "salinity", "qc_policy": "qc_policy_v1", "temporal_filter": "nearest_or_exact",
            "depth_min": 5, "depth_max": 50, "groups": groups}})
        seen = {g["label"]: (g["profiles"], g["profiles_rejected_qc"], g["excluded_temporal_status"], g["levels_with_pairs"])
                for g in out["groups"]}
        self.assertEqual(seen, {"03": (4, 1, 0, 14), "04": (2, 0, 2, 0), "05": (0, 0, 0, 0)})
        stored = episode.results[out["result"]]["groups"][2]["depths"]
        self.assertTrue(all(r["n"] == 0 and r["bias"] is None for r in stored))

    def test_partial_keeps_empty_group_and_judges_refusal(self):
        spec = spec_for(self.data, [["2018-03"], ["2018-05"]], ["mld"])
        self.assertEqual(spec["reference"]["status"], "partial")
        trace = self.episode(spec, oracle_texts(spec, self.lab))
        self.assertEqual(trace["metrics"]["success"], 1)
        groups = trace["registry"]["results"]["res_1"]["groups"]
        self.assertEqual([g["profiles"] for g in groups], [4, 0])
        self.assertEqual(groups[1]["counts"], dict.fromkeys(groups[1]["counts"], 0))
        texts = oracle_texts(spec, self.lab)
        for status, failure, refusal in (("complete", "status", 0), ("insufficient", "status", 1)):
            final = json.loads(texts[-1].split("<final>")[1][:-len("</final>")])
            final["status"] = status
            metrics = self.episode(spec, texts[:-1] + ["<final>" + json.dumps(final) + "</final>"])["metrics"]
            self.assertEqual((metrics["success"], metrics["failure"], metrics["false_refusal"]), (0, failure, refusal))

    def test_labels_and_order_are_not_scientific_conditions(self):
        spec = spec_for(self.data, [["2018-03"], ["2018-05"]], ["mld"])
        groups = [{"label": "五月", "start_date": "2018-05-01", "end_date": "2018-05-31"},
                  {"label": "三月", "start_date": "2018-03-01", "end_date": "2018-03-31"}]
        trace = self.episode(spec, [self.query("2018-03-01", "2018-05-31"), self.mld(groups),
                                    self.final("partial", mld="res_1")])
        self.assertEqual(trace["metrics"]["success"], 1, trace["metrics"])
        short = [dict(groups[1], end_date="2018-03-30"), groups[0]]  # same records, wrong calendar coverage
        trace = self.episode(spec, [self.query("2018-03-01", "2018-05-31"), self.mld(short),
                                    self.final("partial", mld="res_1")])
        self.assertEqual(trace["metrics"]["parameter_errors"], {"mld": ["groups"]})

    def test_diagnostic_on_the_way_is_allowed_but_not_as_delivery(self):
        spec = spec_for(self.data, [["2018-03"]], ["mld"])
        march = [{"label": "3", "start_date": "2018-03-01", "end_date": "2018-03-31"}]
        steps = [self.query("2018-03-01", "2018-03-31"), self.mld(march, "finite_only"), self.mld(march)]
        trace = self.episode(spec, steps + [self.final("complete", mld="res_2")])
        diag, formal = (trace["calls"][i]["result"]["groups"][0]["counts"] for i in (1, 2))
        self.assertEqual((diag["profile_excluded"], formal["profile_excluded"]), (0, 1))
        self.assertEqual(trace["metrics"]["success"], 1)
        self.assertEqual(trace["metrics"]["diagnostic_handles"], 1)
        trace = self.episode(spec, steps + [self.final("complete", mld="res_1")])
        self.assertEqual((trace["metrics"]["success"], trace["metrics"]["parameter_errors"]), (0, {"mld": ["qc_policy"]}))

    def test_in_situ_temperature_inspectable_but_not_computable(self):
        episode = self.lab.episode()
        ds = episode.execute({"name": "query_profiles", "arguments": {
            "lat_min": 10, "lat_max": 20, "lon_min": 150, "lon_max": 160,
            "start_date": "2018-03-01", "end_date": "2018-03-31"}})["dataset"]
        info = episode.execute({"name": "inspect_dataset", "arguments": {"dataset": ds, "variable": "in_situ_temperature"}})
        self.assertIn("势温", info["glorys"])
        self.assertEqual(info["profile_qc_counts"], "不可用")
        groups = [{"label": "3", "start_date": "2018-03-01", "end_date": "2018-03-31"}]
        for name, extra in (("compute_mld", {}), ("compute_consistency", {"temporal_filter": "nearest_or_exact",
                                                                          "depth_min": 5, "depth_max": 50})):
            out = episode.execute({"name": name, "arguments": {"dataset": ds, "variable": "in_situ_temperature",
                                                               "qc_policy": "qc_policy_v1", "groups": groups, **extra}})
            self.assertEqual(out["error_kind"], "incompatible_variable")
        self.assertEqual(episode.results, {})

    def test_partition_is_checked_on_the_query_range(self):
        dev = self.lab.episode()
        q = lambda e, lat, lon, start, end: e.execute({"name": "query_profiles", "arguments": {
            "lat_min": lat[0], "lat_max": lat[1], "lon_min": lon[0], "lon_max": lon[1],
            "start_date": start, "end_date": end}})
        for lat, lon, start, end in (((-20, -10), (-10, 0), "2018-01-01", "2018-01-31"),   # touches the 1-degree buffer
                                     ((-30, -25), (-30, -25), "2018-01-01", "2018-01-31"),  # inside the reserved region
                                     ((10, 20), (150, 160), "2019-11-01", "2019-12-02"),    # end inside the year buffer
                                     ((10, 20), (150, 160), "2017-01-30", "2017-02-28")):
            out = q(dev, lat, lon, start, end)
            self.assertEqual(out.get("error_kind"), "partition", (lat, lon, start, end, out))
        self.assertEqual(q(dev, (10, 20), (150, 160), "2017-01-31", "2019-12-01")["partition"], "dev")
        self.assertEqual(split_dates("train")[1], pd.Timestamp("2016-12-01"))
        spatial = OceanLab("test-spatial", self.data).episode()
        self.assertEqual(q(spatial, (-30, -25), (-30, -25), "2021-01-01", "2021-03-31")["partition"], "test-reserved")
        for lat, lon, start, end in (((-40, -30), (-30, -25), "2021-01-01", "2021-03-31"),  # reaches the buffer
                                     ((-30, -25), (-30, -25), "2018-01-01", "2018-03-31"),  # dev years
                                     ((10, 20), (150, 160), "2021-01-01", "2021-03-31")):   # outside the region
            self.assertEqual(q(spatial, lat, lon, start, end).get("error_kind"), "partition")
        with self.assertRaises(ValueError):
            query_partition("test", (-30, -25), (-30, -25), "2021-01-01", "2021-03-31")

    def test_read_result_pages_with_cursor(self):
        episode = self.lab.episode()
        ds = episode.execute({"name": "query_profiles", "arguments": {
            "lat_min": 10, "lat_max": 20, "lon_min": 150, "lon_max": 160,
            "start_date": "2018-03-01", "end_date": "2018-03-31"}})["dataset"]
        res = episode.execute({"name": "compute_mld", "arguments": {
            "dataset": ds, "variable": "potential_temperature", "qc_policy": "qc_policy_v1",
            "groups": [{"label": "m", "start_date": "2018-03-01", "end_date": "2018-03-31"}]}})["result"]
        page = dict(envmod.PAGE)
        envmod.PAGE["compute_mld"] = 3
        try:
            read = lambda **kw: episode.execute({"name": "read_result", "arguments": {"result": res, "group": "m", **kw}})
            first, second = read(), read(cursor=3)
            self.assertEqual((len(first["rows"]), first["next_cursor"], first["done"]), (3, 3, False))
            self.assertIn("provenance", first)
            self.assertEqual((len(second["rows"]), second["next_cursor"], second["done"]), (1, None, True))
            self.assertEqual(read(cursor=4)["error_kind"], "invalid_request")
            self.assertEqual(read(cursor=-1)["error_kind"], "schema")
            self.assertEqual(sorted(r[4] for r in first["rows"] + second["rows"]),
                             ["estimable"] * 3 + ["profile_excluded"])
        finally:
            envmod.PAGE.update(page)

    def test_self_check_reports_environment_not_model_errors(self):
        spec = spec_for(self.data, [["2018-03"]], ["mld"])
        trace = self.episode(spec, oracle_texts(spec, self.lab))
        self.assertEqual(trace["metrics"]["env_error"], 0)
        broken = copy.deepcopy(trace)
        broken["registry"]["results"]["res_1"]["groups"][0]["counts"]["estimable"] += 1
        metrics = self.harness.scorer.score(spec, broken)
        self.assertEqual((metrics["env_error"], metrics["success"]), (1, 1))
        self.assertIn("counts", metrics["env_problems"][0])
        with self.assertRaises(RuntimeError):
            aggregate([{**broken, "metrics": metrics}])

    def test_protocol_violations_cost_reward_not_success(self):
        spec = spec_for(self.data, [["2018-03"]], ["mld"])
        texts = oracle_texts(spec, self.lab)
        bad = tool_text("查询", "query_profiles", {"lat_min": 10, "lat_max": 20})
        for extra in (bad, "<tool_call>{</tool_call>", "我先想想。"):
            with self.subTest(extra):
                trace = self.episode(spec, [extra] + texts)
                self.assertEqual((trace["metrics"]["success"], trace["metrics"]["reward"]), (1, 0.9))
        trace = self.episode(spec, texts[:-1] + ["<final>完成</final>"])
        self.assertEqual((trace["metrics"]["failure"], trace["metrics"]["reward"]), ("final_format", -0.1))
        trace = self.episode(spec, texts[:1] + [tool_text("x", "query_profiles", {**json.loads(
            texts[0].split("<tool_call>")[1][:-12])["arguments"], "start_date": "2016-01-01"})] + texts[1:])
        self.assertEqual(trace["calls"][1]["result"]["error_kind"], "partition")
        self.assertEqual((trace["metrics"]["success"], trace["metrics"]["reward"]), (1, 1))

    def test_feedback_never_contains_angle_brackets(self):
        # Models copy error text into their plans, and a plan with an angle bracket is a protocol violation.
        spec = spec_for(self.data, [["2018-03"]], ["mld"])
        bad = ['<plan>确保 lat_min < lat_max</plan>' + self.query("2018-03-01", "2018-03-31"),
               '<tool_call>{"name": "query_profiles", "arguments": {"lat_min": 10,}}</tool_call>',
               '<final>{"status": "complete"}></final>', '先查询。',
               tool_text("查", "query_profiles", {"lat_min": 20, "lat_max": 10, "lon_min": 150, "lon_max": 160,
                                                 "start_date": "2018-03-01", "end_date": "2018-03-31"}),
               tool_text("查", "query_profiles", {"lat_min": 10, "lat_max": 20, "lon_min": 160, "lon_max": 150,
                                                 "start_date": "2018-03-01", "end_date": "2018-03-31"})]
        trace = self.episode(spec, bad + oracle_texts(spec, self.lab))
        errors = [c["result"]["error"] for c in trace["calls"] if "error" in c["result"]]
        self.assertEqual(len(errors), len(bad))
        for message in errors:
            self.assertNotRegex(message, "[<>]")
        self.assertIn("尖括号", errors[0])
        self.assertIn("JSON 无法解析", errors[1])
        self.assertEqual(trace["metrics"]["success"], 1)

    def test_profile_cap_counts_only_grouped_profiles(self):
        # A long query span is fine (index only); the cap applies to the profiles a computation reads.
        cap = envmod.MAX_PROFILES
        envmod.MAX_PROFILES = 3
        try:
            episode = self.lab.episode()
            ds = episode.execute({"name": "query_profiles", "arguments": {
                "lat_min": 10, "lat_max": 20, "lon_min": 150, "lon_max": 160,
                "start_date": "2017-02-01", "end_date": "2019-11-30"}})
            self.assertEqual(ds["profiles"], 6)
            compute = lambda month, last: episode.execute({"name": "compute_mld", "arguments": {
                "dataset": ds["dataset"], "variable": "potential_temperature", "qc_policy": "qc_policy_v1",
                "groups": [{"label": month, "start_date": f"2018-{month}-01", "end_date": f"2018-{month}-{last}"}]}})
            self.assertEqual(compute("03", 31)["error_kind"], "query_too_large")
            self.assertEqual(compute("04", 30)["groups"][0]["profiles"], 2)
            self.assertEqual(episode.execute({"name": "inspect_dataset", "arguments": {
                "dataset": ds["dataset"], "variable": "salinity"}})["error_kind"], "query_too_large")
        finally:
            envmod.MAX_PROFILES = cap

    def test_request_status_needs_cells(self):
        with self.assertRaises(ValueError):
            request_status({})
        with self.assertRaises(ValueError):
            request_status({"consistency": {"a": {"depths": []}}})

    def test_observation_stays_bounded_and_keeps_handle(self):
        result = {"result": "res_9", "method_card": "profile_consistency_v1",
                  "groups": [{"label": f"g{i}", "rows": [[1.0, 2, 0.1, 0.2, 0.3]] * 30} for i in range(12)]}
        seen = json.loads(observe(result, self.tok, limit=300))
        self.assertEqual(seen["result"], "res_9")
        self.assertTrue(seen["truncated"])
        self.assertNotIn("rows", seen["groups"][0])


@unittest.skipUnless(HAS_SDK and (ROOT / "data/processed/oceanlab/oceanlab_dev_v1.jsonl").exists()
                     and Path("/data/anhuang/oceandepths/SOURCE_MANIFEST.json").exists(), "local OceanDepths copy required")
class RealDevRequestTests(unittest.TestCase):
    def test_oracle_episodes_through_sdk(self):
        tok = AutoTokenizer.from_pretrained(ROOT / "model")
        lab = OceanLab("dev")
        harness = OceanLabHarness(Scorer(lab.data))
        requests = read_jsonl(ROOT / "data/processed/oceanlab/oceanlab_dev_v1.jsonl")
        picked = [next(r for r in requests if r["reference"]["status"] == s and r["family"] == f)
                  for s, f in (("complete", "report"), ("partial", "consistency_temperature"),
                               ("insufficient", "consistency_salinity"), ("partial", "mld"))]
        traces = []
        for spec in picked:
            texts = iter(oracle_texts(spec, lab))
            generate = lambda ids, limit: (tok.encode(next(texts), add_special_tokens=False) + [tok.eos_token_id], [])
            traces.append(run_trajectory(spec, tok, lab, generate, harness=harness))
        summary = aggregate(traces)
        self.assertEqual(summary["success"], 1, [t["metrics"] for t in traces])
        self.assertEqual(summary["reward"], 1)


if __name__ == "__main__":
    unittest.main()
