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

    def test_an_analysis_may_be_delivered_as_several_handles(self):
        spec = spec_for(self.data, [["2018-03"], ["2018-04"]], ["mld"])
        march = [{"label": "3", "start_date": "2018-03-01", "end_date": "2018-03-31"}]
        april = [{"label": "4", "start_date": "2018-04-01", "end_date": "2018-04-30"}]
        steps = [self.query("2018-03-01", "2018-04-30"), self.mld(march), self.mld(april)]
        for results, ok, failure in (({"mld": ["res_1", "res_2"]}, 1, None),
                                     ({"mld": {"三月": "res_1", "四月": "res_2"}}, 1, None),
                                     ({"mld": ["res_1"]}, 0, "parameters"),            # April missing
                                     ({"mld": ["res_1", "res_1", "res_2"]}, 0, "parameters"),  # March twice
                                     ({"mld": []}, 0, "final_format")):
            final = "<final>" + json.dumps({"status": "partial", "results": results}) + "</final>"
            metrics = self.episode(spec, steps + [final])["metrics"]
            self.assertEqual((metrics["success"], metrics["failure"]), (ok, failure), results)
        self.assertEqual(spec["reference"]["status"], "partial")

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


@unittest.skipUnless(HAS_SDK, "optional SDK environment required")
class RequestSetV2Tests(unittest.TestCase):
    """Request set v2 building blocks and the v2 environment (prompt, multiturn replay, handle reuse)."""

    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(ROOT / "model")
        cls.data = FakeData(fake_profiles())
        cls.lab = OceanLab("dev", cls.data)

    def test_seasons_follow_the_hemisphere(self):
        from agent.oceanlab.requests_v2 import season_months, season_phrase
        self.assertEqual(season_months("冬季", "N", 2017), ["2017-12", "2018-01", "2018-02"])
        self.assertEqual(season_months("冬季", "S", 2017), ["2017-06", "2017-07", "2017-08"])
        self.assertEqual(season_months("夏季", "S", 2017), ["2017-12", "2018-01", "2018-02"])
        self.assertEqual(season_phrase("冬季", "N", 2017), "2017/18年冬季")
        self.assertEqual(season_phrase("冬季", "S", 2017), "2017年冬季")

    def test_region_text_keeps_hemisphere_signs(self):
        from agent.oceanlab.requests_v2 import region_text
        box = [[-20, -10], [-130, -120]]
        self.assertEqual(region_text(box, "degree_letters"), "10°S–20°S、120°W–130°W")
        self.assertEqual(region_text(box, "english"), "10-20S, 120-130W")
        self.assertEqual(region_text(box, "center"), "以南纬15度、西经125度为中心，南北各5度、东西各5度的方框")
        self.assertIn("经度-130到-120", region_text(box, "signed"))

    def test_extraction_must_match_the_spec_exactly(self):
        from agent.oceanlab.requests_v2 import mismatches
        spec = {"region": [[-20, -10], [120, 130]], "groups": [{"months": ["2018-03"]}, {"months": ["2018-04"]}],
                "analyses": ["consistency"], "consistency": {"variable": "salinity", "depth_range_m": [5.0, 100.0]}}
        good = {"lat_min": -20, "lat_max": -10, "lon_min": 120, "lon_max": 130, "groups": [["2018-04"], ["2018-03"]],
                "analyses": ["consistency"], "variable": "salinity", "depth_min": 5, "depth_max": 100,
                "direction": None, "extra": ""}
        self.assertEqual(mismatches(spec, good), [])
        cases = {"region": {"lat_min": 10, "lat_max": 20}, "groups": {"groups": [["2018-03", "2018-04"]]},
                 "depth": {"depth_min": 0.49}, "direction": {"direction": "observation_minus_reanalysis"},
                 "extra": {"extra": "再做密度法"}, "variable": {"variable": "potential_temperature"}}
        for field, change in cases.items():
            self.assertEqual(mismatches(spec, {**good, **change}), [field], field)
        self.assertTrue(mismatches(spec, "not json")[0].startswith("unparsable"))

    def test_v2_prompt_adds_conventions_and_keeps_v1(self):
        self.assertIn("南半球冬与夏", envmod.SYSTEM_V2)
        self.assertIn("多轮对话", envmod.SYSTEM_V2)
        self.assertNotIn("多轮对话", envmod.SYSTEM_V1)
        self.assertEqual(OceanLabHarness().system, envmod.SYSTEM_V1)
        self.assertEqual(OceanLabHarness(prompt="v2").system, envmod.SYSTEM_V2)

    def test_multiturn_replays_history_and_allows_handle_reuse(self):
        consistency = {"variable": "potential_temperature", "shallow_requested": False, "depth_range_m": [5.0, 50.0]}
        first = spec_for(self.data, [["2018-03"]], ["consistency"], consistency, sid="fake-first")
        first_trace = run_trajectory(first, self.tok, self.lab, generator_of(self.tok, oracle_texts(first, self.lab)),
                                     harness=OceanLabHarness(prompt="v2"))
        final = spec_for(self.data, [["2018-03"]], ["mld", "consistency"], consistency, sid="fake-final")
        final.update(history=first_trace["messages"][2:] + [{"role": "user", "content": "再加上混合层"}],
                     history_calls=[c["call"] for c in first_trace["calls"]],
                     history_handles=["ds_1", "res_1"])
        harness = OceanLabHarness(Scorer(self.data), prompt="v2")
        reuse = [tool_text("复用ds_1", "compute_mld", {"dataset": "ds_1", "variable": "potential_temperature",
                                                       "qc_policy": "qc_policy_v1", "groups": [
                                                           {"label": "3", "start_date": "2018-03-01", "end_date": "2018-03-31"}]}),
                 '<final>{"status": "complete", "results": {"mld": "res_2", "consistency": "res_1"}}</final>']
        trace = run_trajectory(final, self.tok, self.lab, generator_of(self.tok, reuse), harness=harness)
        self.assertEqual(trace["metrics"]["success"], 1, trace["metrics"])
        self.assertEqual(trace["calls"][0]["result"]["result"], "res_2")
        prompt = self.tok.decode(trace["rounds"][0]["input_ids"])
        self.assertIn("再加上混合层", prompt)
        self.assertIn(envmod.SYSTEM_V2, prompt)
        with self.assertRaises(RuntimeError):
            harness.start({**final, "history_handles": ["ds_9", "res_1"]}, self.lab)

    def test_skill_preface_is_retrieved_and_removable(self):
        from agent.oceanlab.skills import SkillMemory
        skill = lambda sid, text: {"id": sid, "title": sid, "principle": text, "when": "", "evidence": []}
        bank = {"general": [skill("g-1", "先查询再计算")], "common_mistakes": [skill("cm-1", "深度 <5.078 m 不算")],
                "task_specific": {"mld": [skill("mld-1", "混合层按月分组")], "season": [skill("season-1", "季节按半球")],
                                  "consistency": [skill("con-1", "看偏差方向")]}}
        spec = spec_for(self.data, [["2018-03"]], ["mld"])
        spec["text"] = "请算一下这个区域冬季的混合层"
        harness = OceanLabHarness(Scorer(self.data), prompt="v2", memory=SkillMemory(bank))
        trace = run_trajectory(spec, self.tok, self.lab, generator_of(self.tok, oracle_texts(spec, self.lab)),
                               harness=harness)
        self.assertEqual(trace["memory"]["items"], ["g-1", "mld-1", "season-1", "cm-1"])
        first = trace["messages"][1]["content"]
        preface = first[:trace["memory"]["preface_chars"]]
        self.assertEqual(first[len(preface):], spec["text"])
        self.assertNotIn("<", preface)
        self.assertIn("小于5.078", preface)
        self.assertEqual(trace["messages"][0]["content"], envmod.SYSTEM_V2)


def generator_of(tok, texts):
    sequence = iter(texts)
    return lambda ids, limit: (tok.encode(next(sequence), add_special_tokens=False) + [tok.eos_token_id], [])


@unittest.skipUnless(HAS_SDK, "optional SDK environment required")
class BatchedRolloutTests(unittest.TestCase):
    """Agent RL rollouts: concurrent SDK episodes served by batched generation, and the group advantages."""

    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(ROOT / "model")
        cls.data = FakeData(fake_profiles())
        cls.lab = OceanLab("dev", cls.data)

    def scripted_batches(self, scripts, sizes):
        queues = {i: iter(texts) for i, texts in scripts.items()}

        def generate_batch(requests):
            sizes.append(len(requests))
            return [self.tok.encode(next(queues[i]), add_special_tokens=False) + [self.tok.eos_token_id]
                    for i, _, _ in requests]
        return generate_batch

    def test_batched_episodes_equal_sequential_ones(self):
        from functools import partial
        from trainer.oceanlab_rollout import BatchedEpisodes
        harness = OceanLabHarness(Scorer(self.data), prompt="v2")
        specs = [spec_for(self.data, [["2018-03"]], ["mld"], sid=f"fake-{i}") for i in range(3)]
        scripts = {i: oracle_texts(spec, self.lab) for i, spec in enumerate(specs)}
        scripts[1] = ["先想一想"] + scripts[1]  # one invalid action: this episode needs an extra round
        scripts[2] = scripts[2][:1] + ['<final>{"status": "complete", "results": {}}</final>']
        sequential = [run_trajectory(spec, self.tok, self.lab, generator_of(self.tok, scripts[i]), harness=harness)
                      for i, spec in enumerate(specs)]
        sizes = []
        batched = BatchedEpisodes(self.scripted_batches(scripts, sizes)).run(
            [partial(run_trajectory, spec, self.tok, self.lab, harness=harness) for spec in specs])
        strip = lambda t: {**t, "rounds": [{k: v for k, v in r.items() if k not in ("seconds", "sdk_call_id")}
                                           for r in t["rounds"]]}
        self.assertEqual([strip(t) for t in batched], [strip(t) for t in sequential])
        self.assertEqual(sizes[0], 3)
        self.assertEqual(sum(sizes), sum(len(t["rounds"]) for t in sequential))
        self.assertEqual([t["metrics"]["reward"] for t in batched], [1.0, 0.9, 0.0])  # S=1 for the second, minus 0.1 protocol

    def test_an_episode_error_is_raised_after_the_others_end(self):
        from trainer.oceanlab_rollout import BatchedEpisodes
        finished = []

        def good(generate):
            tokens, _ = generate([1, 2], 4)
            finished.append(tokens)
            return tokens

        def bad(generate):
            generate([3], 4)
            raise KeyError("episode failed")
        with self.assertRaises(KeyError):
            BatchedEpisodes(lambda requests: [[7]] * len(requests)).run([good, bad, good])
        self.assertEqual(finished, [[7], [7]])

    def test_concurrent_episodes_keep_their_own_memory_record(self):
        from functools import partial
        from agent.oceanlab.skills import SkillMemory
        from trainer.oceanlab_rollout import BatchedEpisodes
        skill = lambda sid: {"id": sid, "title": sid, "principle": "规则", "when": "", "evidence": []}
        bank = {"general": [], "common_mistakes": [],
                "task_specific": {"mld": [skill("mld-1")], "season": [skill("season-1")]}}
        harness = OceanLabHarness(Scorer(self.data), prompt="v2", memory=SkillMemory(bank))
        specs = [spec_for(self.data, [["2018-03"]], ["mld"], sid=f"fake-{i}") for i in range(2)]
        specs[0]["text"], specs[1]["text"] = "算一下混合层", "冬季的情况"
        scripts = {i: oracle_texts(spec, self.lab) for i, spec in enumerate(specs)}
        traces = BatchedEpisodes(self.scripted_batches(scripts, [])).run(
            [partial(run_trajectory, spec, self.tok, self.lab, harness=harness) for spec in specs])
        self.assertEqual([t["memory"]["items"] for t in traces], [["mld-1"], ["season-1"]])

    def test_skill_evolution_adds_reviewed_skills_under_caps(self):
        from agent.oceanlab import evolve as ev
        stats = {"mld": (30, 0.35), "season": (12, 0.55), "relative_time": (5, 0.1), "consistency": (40, 0.7)}
        self.assertEqual(ev.weak_categories(stats, 0.6, 0.4, 10, 2), ["mld"])
        self.assertEqual(ev.weak_categories(stats, 0.6, 0.4, 10, 2, relative=True), ["mld", "season"])
        traces = [{"id": f"r{i}", "metrics": {"success": 0.0, "failure": kind}}
                  for i, kind in enumerate(["parameters"] * 5 + ["status", "unknown_handle"])]
        picked = ev.diverse_failures(traces + [{"id": "ok", "metrics": {"success": 1.0, "failure": None}}], 3, 0)
        self.assertEqual(sorted(t["metrics"]["failure"] for t in picked), ["parameters", "status", "unknown_handle"])

        class FakeTeacher:
            def __init__(self, replies):
                self.replies = iter(replies)

            def chat(self, messages, **kwargs):
                self.prompt = messages[-1]["content"]
                return json.dumps(next(self.replies), ensure_ascii=False)
        skill = lambda title: {"title": title, "principle": f"{title}的做法", "when": "", "evidence": ["r0"]}
        bank = {"general": [], "common_mistakes": [{"id": "cm-1", "title": "旧", "principle": "旧规则 <5 m", "when": "",
                                                    "evidence": []}], "task_specific": {"mld": []}}
        reply = {"task_specific": [skill("甲"), skill("乙")], "common_mistakes": [skill("丙"), skill("丁"), skill("戊")]}
        keep = {"keep": [{"index": 1}, {"index": 2, "principle": "乙的准确做法"}, {"index": 4}, {"index": 4}, {"index": 9},
                         {"index": 5}], "removed": [{"index": 3, "reason": "重复"}]}
        teacher = FakeTeacher([reply, keep])
        new, log = ev.evolve(bank, "mld", [(traces[0], {})], lambda t, s: "请求：…", teacher, "", "e25")
        self.assertEqual([s["id"] for s in new["task_specific"]["mld"]], ["e25-mld-1", "e25-mld-2"])
        self.assertEqual([s["id"] for s in new["common_mistakes"]], ["cm-1", "e25-mld-3"])  # at most 3 new skills
        self.assertEqual([s["title"] for s in new["common_mistakes"]], ["旧", "丁"])  # removed and repeated numbers skipped
        self.assertEqual(new["task_specific"]["mld"][1]["principle"], "乙的准确做法")
        self.assertEqual(bank["task_specific"]["mld"], [])  # the old version is unchanged
        self.assertEqual(len(log["added"]), 3)
        self.assertNotIn("<", teacher.prompt.split("新增技能")[0])  # current skills reach the teacher rendered
        full = {**bank, "task_specific": {"mld": [{"id": f"m{i}", "title": "t", "principle": "p"} for i in range(6)]}}
        keep_all = {"keep": [{"index": i} for i in range(1, 6)]}
        new, log = ev.evolve(full, "mld", [(traces[0], {})], lambda t, s: "", FakeTeacher([reply, keep_all]), "", "e50")
        self.assertEqual(len(new["task_specific"]["mld"]), 6)  # category cap reached: only common mistakes are added
        self.assertEqual([a["level"] for a in log["added"]], ["common_mistakes"] * 3)

    def test_candidate_skills_are_verified_alone(self):
        from agent.oceanlab import evolve as ev
        from agent.oceanlab.skills import retrieve
        bank = {"general": [], "common_mistakes": [], "task_specific": {"mld": [{"id": "m1", "title": "a", "principle": "p"}]}}
        trial = ev.with_skill(bank, "mld", {"title": "新", "principle": "新规则", "level": "task_specific"})
        self.assertEqual([s["id"] for s in retrieve(trial, ["混合层"])], ["m1", "candidate"])
        self.assertEqual(len(bank["task_specific"]["mld"]), 1)
        baseline = {"f1": 0.0, "f2": 0.0, "s1": 1.0, "s2": 1.0}
        self.assertEqual(ev.verdict(baseline, {"f1": 1.0, "f2": 0.0, "s1": 1.0, "s2": 1.0}), (1, 0, True))
        self.assertEqual(ev.verdict(baseline, {"f1": 1.0, "f2": 0.0, "s1": 0.0, "s2": 1.0}), (1, 1, False))
        self.assertEqual(ev.verdict(baseline, dict(baseline)), (0, 0, False))  # fixes nothing: not kept
        self.assertEqual(ev.verdict(baseline, {"f1": 1.0, "f2": 1.0, "s1": 0.0, "s2": 1.0}, min_gain=2), (2, 1, True))
        new, added = ev.add_skills(bank, "season", [{"title": "t", "principle": "p", "level": "task_specific"}] * 4, "e5")
        self.assertEqual([a["id"] for a in added], ["e5-season-1", "e5-season-2", "e5-season-3"])
        self.assertNotIn("season", bank["task_specific"])

    def test_group_advantages(self):
        from trainer.train_oceanlab_grpo import group_advantages
        self.assertIsNone(group_advantages([1.0] * 8, "grpo"))
        self.assertIsNone(group_advantages([-0.1, -0.1], "rloo"))
        grpo = group_advantages([1.0, 0.0, 0.0, 0.0], "grpo")
        self.assertAlmostEqual(grpo[0], 0.75 / (0.1875 ** .5 + 1e-4))
        self.assertAlmostEqual(sum(grpo), 0.0)
        for got, want in zip(group_advantages([1.0, 0.0, 0.0, -0.1], "rloo"),
                             [1.0 - (-0.1 / 3), 0.0 - 0.9 / 3, 0.0 - 0.9 / 3, -0.1 - 1.0 / 3]):
            self.assertAlmostEqual(got, want)


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
