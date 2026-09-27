#!/usr/bin/env python3
"""Deterministic, executed ReAct demonstrations; no judge/API or new downloads."""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from agent.ocean import calculate, file_hash, read_jsonl, TOOLS
from agent.react import VERSION, KINDS, ReactTools, run_trajectory
from scripts.prepare_ocean_data import atomic_jsonl, normalized, stable_score, percentile


def digest(value):
    return hashlib.sha256(normalized(value).encode()).hexdigest()


def tool(name, arguments, reason):
    return f"<plan>{reason}</plan><tool_call>" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "</tool_call>"


def oracle(task):
    """Offline teacher only. Never called by the inference harness."""
    texts = []
    if task.get("query"):
        texts.append(tool("search_ocean", {"query": task["query"]}, "先查找相关海洋资料。"))
    for calc in task.get("required_calcs", []):
        text = tool("marine_calculate", calc, "按已知数值与单位计算。")
        texts.append(text)
        if task.get("public", {}).get("transient"):
            texts.append(tool("marine_calculate", calc, "工具暂时不可用，重试一次。"))
    if task["kind"] == "clarify":
        texts.append("<clarify>" + task["answer"] + "</clarify>")
    else:
        answer = f"{task['expected']['value']:g} {task['expected']['unit']}" if "expected" in task else task["answer"]
        if task.get("doc_id"):
            answer += f"[{task['doc_id']}]"
        texts.append("<final>" + answer + "</final>")
    return texts


def prepare(source, output):
    if (output / "agent_manifest.json").exists():
        raise FileExistsError("Use a new output directory; generated data is immutable")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    rng = random.Random(42)
    old_sources, used_sources, used_params, used_questions = set(), set(), set(), set()
    for split in ("train", "val", "test"):
        for d in read_jsonl(source / f"ocean_agent_corpus_{split}.jsonl"):
            if d.get("source_id"):
                old_sources.add(d["source_id"])
    manifest = {"version": VERSION, "seed": 42, "source": "zjunlp/OceanInstruct-v0.2", "source_license": "MIT",
                "warning": "OceanInstruct may contain synthetic/errors. Voyage cards, outages and tasks are synthetic; not real navigation advice.",
                "tool_sha256": file_hash(ROOT / "agent/ocean.py"), "harness_sha256": file_hash(ROOT / "agent/react.py"),
                "splits": {}}
    # Distinct phrasing families, not merely random rows from the same template.
    wrappers = {"train": "任务：{}", "val": "海洋调查记录处理：{}", "test": "请完成以下海上作业资料核对，并遵循输出要求：{}"}
    for split, size in (("train", 3000), ("val", 150), ("test", 300)):
        n = size // 6
        docs, tasks, filtered = [], [], Counter()
        rows = sorted(read_jsonl(source / f"ocean_sft_{split}.jsonl"), key=lambda r: stable_score(r["conversations"][1]["content"], 42))
        selected = []
        for row in rows:
            q, a = [m["content"] for m in row["conversations"] if m["role"] != "system"]
            sid = digest(q)
            excerpt = next((s.strip() for s in re.split(r"(?<=[。！？])\s*|\n", a)
                            if 20 <= len(s.strip()) <= 65 and len(re.findall(r"[\u4e00-\u9fff]", s)) >= 12
                            and s.strip().endswith(("。", "！", "？"))), "")
            if sid in old_sources or sid in used_sources:
                filtered["old_or_duplicate_source"] += 1
                continue
            if not excerpt or not 3 <= len(q) <= 90 or any(c in q + excerpt for c in ("<", ">", "\ufffd")):
                filtered["ineligible_excerpt"] += 1
                continue
            used_sources.add(sid)
            selected.append({"id": "O" + sid[:10], "source_id": sid, "title": q,
                             "excerpt": excerpt, "source": "OceanInstruct-v0.2"})
            if len(selected) == n * 2:
                break
        if len(selected) != n * 2:
            raise ValueError(f"Insufficient eligible source documents in {split}")
        for i, doc in enumerate(selected):
            kind = "retrieve" if i < n else "direct"
            task = {"kind": kind, "source_id": doc["source_id"], "answer": doc["excerpt"], "required_calcs": []}
            if kind == "retrieve":
                docs.append(doc)
                question = {"train": f"关于‘{doc['title']}’，请提供资料库中的原文证据，末尾标注[文档编号]。",
                            "val": f"请为问题‘{doc['title']}’找出资料中的一句原文，并附上[文档编号]。",
                            "test": f"待核对的问题是：{doc['title']}。答复须由资料原文和[文档编号]组成，不要改写。"}[split]
                task.update(query=doc["title"], doc_id=doc["id"], required_docs=[doc["id"]])
            else:
                question = {"train": f"下面是已核实的海洋记录：{doc['excerpt']}\n只原样返回这段记录，不需要补充资料。",
                            "val": f"请照录这份已确认海洋记录的内容：{doc['excerpt']} 不要添加别的文字。",
                            "test": f"这是全部需要使用的资料：【{doc['excerpt']}】。请只输出括号内的原文，不作外部查询。"}[split]
            task["question"] = question
            tasks.append(task)
        for kind in ("calculate", "chain", "recovery", "clarify"):
            for i in range(n):
                while True:
                    speed, hours, distance = rng.randint(40, 400) / 10, rng.randint(1, 120) / 10, rng.randint(10, 3000) / 10
                    op = ("convert", "distance", "time", "speed")[i % 4] if kind == "calculate" else "distance"
                    if op == "convert":
                        calc = dict(operation=op, value=distance, from_unit="nmi", to_unit="km")
                        q = f"海洋调查航程为{distance:g} nmi，相当于多少km？只回答数值和单位。"
                    elif op == "distance":
                        calc = dict(operation=op, speed=speed, hours=hours)
                        q = f"调查船航速为{speed:g} km/h，持续{hours:g} h，航程是多少？只回答数值和单位。"
                    elif op == "time":
                        calc = dict(operation=op, distance=distance, speed=speed)
                        q = f"航速{speed:g} km/h的调查船走完{distance:g} km需要多久？只回答数值和单位。"
                    else:
                        calc = dict(operation=op, distance=distance, hours=hours)
                        q = f"海洋调查船{hours:g} h行驶{distance:g} km，平均航速是多少？只回答数值和单位。"
                    if split != "train":
                        values = {"convert": (f"航程表记载{distance:g} nmi，请以km表示这段航程。", f"要填写km制报表，原始海里值是{distance:g} nmi，填写什么数值？"),
                                  "distance": (f"已知航速{speed:g} km/h与航行时长{hours:g} h，求调查船走过的距离。", f"某调查船保持{speed:g} km/h匀速经过{hours:g} h，请给出累计航程。"),
                                  "time": (f"调查航程{distance:g} km，船以{speed:g} km/h匀速行驶，求所需小时数。", f"以{speed:g} km/h完成{distance:g} km的海上任务，耗时应填多少h？"),
                                  "speed": (f"某海上任务总长{distance:g} km，总时长{hours:g} h，求km/h表示的航速。", f"从记录得知船舶走过{distance:g} km并用时{hours:g} h，其平均速度为多少？")}
                        q = values[op][0 if split == "val" else 1] + "只回答数值和单位。"
                    key = json.dumps(calc, sort_keys=True)
                    if key not in used_params:
                        used_params.add(key)
                        break
                tid = digest(key)[:10]
                task = {"kind": kind, "question": q, "parameter_key": key, "synthetic": True,
                        "required_calcs": [calc], "expected": calculate(calc)}
                if kind == "chain":
                    cid = "C" + tid
                    title = f"模拟航次卡 {cid}"
                    speed_kn = speed
                    conversion = dict(operation="convert", value=speed_kn, from_unit="kn", to_unit="km/h")
                    converted = calculate(conversion)["value"]
                    final_calc = dict(operation="distance", speed=converted, hours=hours)
                    docs.append({"id": cid, "title": title, "source": "synthetic-v2",
                                 "excerpt": f"合成练习，非实测：speed={speed_kn:g} kn，hours={hours:g} h。"})
                    chain_question = {"train": f"{title}对应的调查船航程是多少km？",
                                      "val": f"请根据{title}的航速和时长确定总里程，统一以km表示。",
                                      "test": f"需要补全km制航程报表，已知依据仅有{title}，请给出应填写的航程。"}[split]
                    task.update(question=chain_question + "只回答数值和单位。",
                                query=title, required_docs=[cid], required_calcs=[conversion, final_calc], expected=calculate(final_calc))
                elif kind == "recovery":
                    if i % 2 == 0:
                        task["public"] = {"transient": True}
                    else:
                        wrong = dict(calc, operation="speed")  # real schema error: speed instead of distance
                        call = {"name": "marine_calculate", "arguments": wrong}
                        error = ReactTools([]).execute(call)
                        assert "error" in error
                        task["public"] = {"history": [{"role": "assistant", "content": tool("marine_calculate", wrong, "计算航程。")},
                                                      {"role": "tool", "content": json.dumps(error, ensure_ascii=False)}]}
                elif kind == "clarify":
                    task.pop("expected")
                    task["required_calcs"] = []
                    if i % 3 == 0:
                        task.update(question=f"调查船持续航行{hours:g} h，请给出航程。船舶编号N{tid}。",
                                    answer="请补充调查船的航速及单位。", clarify_keywords=["航速", "单位"], no_tools=True)
                    elif i % 3 == 1:
                        query = "missing" + tid
                        task.update(question=f"编号{query}的航次记录写了什么？缺少资料时请说明需要补充什么。",
                                    query=query, require_empty_search=True,
                                    answer="未找到航次资料，请补充正确的航次编号或原始资料。", clarify_keywords=["资料", "补充"])
                    else:
                        title, cid = f"模拟冲突航次卡 X{tid}", "X" + tid
                        for suffix, value in (("a", speed), ("b", speed + 5)):
                            docs.append({"id": cid + suffix, "title": title, "source": "synthetic-v2",
                                         "excerpt": f"合成练习，同日未核实记录：speed={value:g} km/h，hours={hours:g} h。"})
                        task.update(question=f"{title}的船舶航程是多少？若记录不一致，请先确认。", query=title,
                                    required_docs=[cid + "a", cid + "b"], answer="资料中的航速存在冲突，请确认应采用哪条记录。",
                                    clarify_keywords=["航速", "冲突", "确认"])
                tasks.append(task)
        # Clarification scenarios use held-out phrasing too; gold fields remain private.
        for task in tasks:
            if task["kind"] == "clarify" and split != "train":
                if task.get("no_tools"):
                    task["question"] = (
                        f"船舶记录只有时长，未记载速度。记录编号{digest(task['parameter_key'])[:10]}，目前能确定航程吗？" if split == "val" else
                        f"请为编号{digest(task['parameter_key'])[:10]}的调查航次补填航程。日志只提供航行时长，尚无船速数据。")
                elif task.get("require_empty_search"):
                    task["question"] = (f"请找出航次{task['query']}的资料；无法定位时请告诉我需补充的信息。" if split == "val" else
                                        f"需要核验{task['query']}这条航次记录的内容；查无记录时请向我索取依据。")
                else:
                    task["question"] = (f"请核对{task['query']}的记录。如果不同记录不能支持唯一航速，应先向我确认。" if split == "val" else
                                        f"以{task['query']}为依据填写航程；不允许自行选择互相矛盾的数据，请说明待确认事项。")
        env = ReactTools(docs)
        demos, lengths = [], []
        tasks.sort(key=lambda t: stable_score(t["question"], 42))
        for task in tasks:
            task.update(version=VERSION, template_family=split + ":" + task["kind"])
            task["question"] = wrappers[split].format(task["question"])
            task["id"] = split + "-" + digest(task["question"])[:16]
            fingerprint = normalized(task["question"])
            if fingerprint in used_questions:
                raise ValueError("duplicate prompt")
            used_questions.add(fingerprint)
            texts = iter(oracle(task))
            def generate(ids, limit):
                output_ids = tokenizer.encode(next(texts), add_special_tokens=False) + [tokenizer.eos_token_id]
                if len(output_ids) > limit:
                    raise ValueError(f"Overlong target: {task['id']}")
                return output_ids, []
            trace = run_trajectory(task, tokenizer, env, generate)
            if trace["metrics"]["success"] != 1:
                raise ValueError(f"Teacher failed real execution: {task['id']} {trace}")
            messages = trace["messages"]
            messages[0]["tools"] = json.dumps(TOOLS, ensure_ascii=False)
            supervision_start = 2 + len(task.get("public", {}).get("history", []))
            demos.append({"conversations": messages, "supervision_start": supervision_start})
            length = len(tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True))
            if length > 2048:
                raise ValueError("Overlong demonstration; no silent truncation")
            lengths.append(length)
        outputs = {f"ocean_agent_{split}.jsonl": tasks, f"ocean_agent_sft_{split}.jsonl": demos,
                   f"ocean_agent_corpus_{split}.jsonl": docs}
        for name, rows in outputs.items():
            atomic_jsonl(output / name, rows)
        manifest["splits"][split] = {"counts": dict(Counter(t["kind"] for t in tasks)), "filtered": dict(filtered),
                                    "source_sha256": file_hash(source / f"ocean_sft_{split}.jsonl"),
                                    "artifacts": {name: file_hash(output / name) for name in outputs},
                                    "tokens": {"p50": percentile(lengths, .5), "p95": percentile(lengths, .95),
                                               "max": max(lengths), "truncation_rate": 0}}
    from trainer.train_agent import atomic_json
    atomic_json(output / "agent_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/react-v2")
    args = parser.parse_args()
    prepare(args.source, args.output)
