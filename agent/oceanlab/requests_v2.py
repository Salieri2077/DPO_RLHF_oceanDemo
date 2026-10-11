"""Request set v2: real-user wording and request combinations that the C0 demonstrations never show (M2b).

Declared before any model is evaluated on them, and never selected by model outcomes:
- structured specs are sampled by the program (same blocks, months and partitions as v1), in three request types:
  paraphrase (the v1 period forms), reasoning (local season by hemisphere, months within a local season, a month
  versus the same month a year earlier) and multiturn (a v1-form request answered by the oracle, then one follow-up:
  switch variable, narrow to one month, split into months, change depth, add the mixed layer, move the region);
- the request text is written by a teacher model imitating a sampled user persona and style, told which facts the
  request must convey and which v1 template phrasings to avoid;
- an independent extraction pass (same teacher, separate prompt, public conventions only, no access to the spec)
  must recover exactly the spec's region, groups, analyses, variable and depth range with no extra requirement;
  otherwise the text is rewritten with another persona (at most four attempts). Only verified texts are kept.
"""
import json
import random

import numpy as np
import pandas as pd

from agent.oceanlab.data import QC_POLICY, REVISION
from agent.oceanlab.tasks import block_box, dev_blocks, month_zh, split_months

TYPE_SHARE = {"paraphrase": .5, "reasoning": .25, "multiturn": .25}
FAMILY_SHARE = {"mld": .375, "consistency_temperature": .21, "consistency_salinity": .165, "report": .25}
V1_KINDS = {"mld": ["monthly_season", "two_months", "season"],
            "consistency_temperature": ["single", "two_months", "season"],
            "consistency_salinity": ["single", "two_months", "season"], "report": ["single", "season"]}
REASONING_KINDS = {"mld": ["local_season", "season_monthly", "same_month_prev_year"],
                   "consistency_temperature": ["local_season", "same_month_prev_year"],
                   "consistency_salinity": ["local_season", "same_month_prev_year"],
                   "report": ["local_season", "same_month_prev_year"]}
# Meteorological seasons: (northern-hemisphere months, southern-hemisphere months).
SEASONS = {"冬季": ((12, 1, 2), (6, 7, 8)), "春季": ((3, 4, 5), (9, 10, 11)),
           "夏季": ((6, 7, 8), (12, 1, 2)), "秋季": ((9, 10, 11), (3, 4, 5))}
DEPTHS = [50, 100, 200, 500]
VARIABLE_ZH = {"potential_temperature": "势温", "salinity": "盐度"}

PERSONAS = ["物理海洋方向的研究生", "做海气相互作用的研究员", "海洋预报中心的值班预报员", "气候模式开发工程师",
            "渔业公司的技术员", "海上风电项目的环境工程师", "高校海洋学老师（在备课）", "科普作者",
            "海军气象水文保障人员", "刚接触 Argo 数据的本科生", "数据同化方向的博士后", "海洋环境监测站的工程师"]
TONES = ["简短口语，像在即时通讯里随手发的", "正式书面，像写给合作方的邮件", "先交代一两句研究背景再提需求",
         "有点着急，语气直接", "分条列出要点", "中英混杂，夹杂 MLD、bias、RMSE、Argo、GLORYS 等英文术语",
         "客气礼貌、句子偏长", "像在组会上口头布置任务"]
COORDINATE_STYLES = ["degree_letters", "spoken", "signed", "center", "short_zh", "english"]


def _ns(v):
    return f"{'北纬' if v >= 0 else '南纬'}{abs(v):g}"


def _ew(v):
    return f"{'东经' if v >= 0 else '西经'}{abs(v):g}"


def region_text(region, style):
    """The region in one of several real-world notations, converted by the program (writers copy it verbatim)."""
    (la0, la1), (lo0, lo1) = region
    lat_lo, lat_hi = sorted((abs(la0), abs(la1))) if la1 <= 0 else (abs(la0), abs(la1))
    lon_lo, lon_hi = sorted((abs(lo0), abs(lo1))) if lo1 <= 0 else (abs(lo0), abs(lo1))
    ns = "N" if la0 >= 0 else "S"
    ew = "E" if lo0 >= 0 else "W"
    nsz, ewz = ("北纬" if la0 >= 0 else "南纬"), ("东经" if lo0 >= 0 else "西经")
    crosses_lat, crosses_lon = la0 < 0 < la1, lo0 < 0 < lo1
    if style == "signed" or crosses_lat or crosses_lon:
        return f"纬度{la0:g}到{la1:g}、经度{lo0:g}到{lo1:g}（负数表示南纬、西经）"
    if style == "degree_letters":
        return f"{lat_lo:g}°{ns}–{lat_hi:g}°{ns}、{lon_lo:g}°{ew}–{lon_hi:g}°{ew}"
    if style == "spoken":
        return f"{nsz}{lat_lo:g}到{lat_hi:g}度、{ewz}{lon_lo:g}到{lon_hi:g}度之间"
    if style == "center":
        cla, clo = (la0 + la1) / 2, (lo0 + lo1) / 2
        return f"以{_ns(cla)}度、{_ew(clo)}度为中心，南北各{(la1 - la0) / 2:g}度、东西各{(lo1 - lo0) / 2:g}度的方框"
    if style == "short_zh":
        return f"{nsz}{lat_lo:g}-{lat_hi:g}°，{ewz}{lon_lo:g}-{lon_hi:g}°"
    return f"{lat_lo:g}-{lat_hi:g}{ns}, {lon_lo:g}-{lon_hi:g}{ew}"


V1_PHRASES = ["请统计", "温度阈值混合层深度，按", "用温度阈值法（10m参考、0.2℃）诊断", "我需要", "先逐剖面计算再汇总",
              "比较EN4与GLORYS的", "请核验GLORYS再分析与EN4观测", "观测—再分析对比", "按层输出 bias / MAE / RMSE / n",
              "说明样本覆盖是否支持", "逐层给出偏差、平均绝对误差、均方根误差和有效配对数"]

CONVENTIONS = """解析约定（与分析助手公开的规则一致）：
- 区域：方框 lat_min<lat_max、lon_min<lon_max；南纬、西经为负数（S、W 同理）。若以中心点加范围描述，换算成方框边界。
- 时段：groups 是若干组，每组是 "YYYY-MM" 月份列表（按时间排序）。要求分别统计或相互对比的每个时段各为一组；作为一个整体统计的连续时段合为一组。
- 季节按区域所在半球的气象季节：北半球冬季=12、1、2月，春季=3～5月，夏季=6～8月，秋季=9～11月；南半球冬季=6～8月，春季=9～11月，夏季=12、1、2月，秋季=3～5月。跨年的季节写作“2017/18年冬季”，指2017年12月至2018年2月。只说季节而没说分月时，整个季节为一组。
- “去年同月”“前一年同期”指前一年的同一个月，它和当月各为一组。
- analyses：混合层深度为 "mld"；观测（EN4）与再分析（GLORYS）一致性为 "consistency"。
- variable：一致性分析的变量，温度/势温为 "potential_temperature"，盐度为 "salinity"；没有一致性分析时为 null。
- 深度（只对一致性分析）：说“上层X米”“X米以浅”“前X米”等而没有明确要求从最浅的0.49米层开始时，depth_min=5、depth_max=X；明确要求从0.49米（最表层）开始时 depth_min=0.49。没有一致性分析时两者为 null。
- direction：一致性差值的方向，请求说“再分析减观测”或等价说法（如 GLORYS 减 EN4）为 "reanalysis_minus_observation"，说“观测减再分析”或等价说法为 "observation_minus_reanalysis"，没说明为 null。
- extra：请求中超出上述字段的额外分析要求，例如另一个区域、变量、时段或产品，密度法等别的方法，敏感性对比；没有则为空字符串。下面这些都是两项分析的标准输出，不算额外要求：混合层的可估计剖面数、中位数、四分位数、无法给出精确深度的剖面数及原因、温度阈值法本身；一致性的再分析减观测、逐层 bias/MAE/RMSE/配对数、样本是否充足；对各时段结果的比较或说明差异。研究背景、语气、对汇报形式的一般要求也不算。"""


def season_months(season, hemisphere, year):
    """Months of a meteorological season. For a season spanning the year end, `year` is the year of its December."""
    months = SEASONS[season][0 if hemisphere == "N" else 1]
    if months == (12, 1, 2):
        return [f"{year}-12", f"{year + 1}-01", f"{year + 1}-02"]
    return [f"{year}-{m:02d}" for m in months]


def season_phrase(season, hemisphere, year):
    crosses = SEASONS[season][0 if hemisphere == "N" else 1] == (12, 1, 2)
    return f"{year}/{str(year + 1)[-2:]}年{season}" if crosses else f"{year}年{season}"


def group(months, label_zh):
    label = months[0] if len(months) == 1 else f"{months[0]}..{months[-1]}"
    return {"label": label, "label_zh": label_zh, "months": months}


def v1_groups(rng, kind, months):
    """The v1 period forms, with a writer-facing description."""
    if kind == "season":
        start = rng.randrange(len(months) - 2)
        ms = [str(m) for m in months[start:start + 3]]
        return [group(ms, f"{month_zh(months[start])}至{month_zh(months[start + 2])}")], \
            f"{month_zh(months[start])}到{month_zh(months[start + 2])}这三个月作为一个整体统计"
    if kind == "two_months":
        a, b = sorted(rng.sample(range(len(months)), 2))
        return [group([str(months[i])], month_zh(months[i])) for i in (a, b)], \
            f"{month_zh(months[a])}和{month_zh(months[b])}两个月，分别统计并相互对比"
    if kind == "monthly_season":
        start = rng.randrange(len(months) - 2)
        return [group([str(m)], month_zh(m)) for m in months[start:start + 3]], \
            f"{month_zh(months[start])}到{month_zh(months[start + 2])}，每个月分别统计"
    i = rng.randrange(len(months))
    return [group([str(months[i])], month_zh(months[i]))], f"{month_zh(months[i])}（单个月）"


def reasoning_groups(rng, kind, months, hemisphere):
    """Periods that need one inference step: local season, its months, or the same month a year earlier."""
    available = {str(m) for m in months}
    if kind == "same_month_prev_year":
        candidates = [m for m in months if str(m - 12) in available]
        m = rng.choice(candidates)
        return [group([str(m - 12)], month_zh(m - 12)), group([str(m)], month_zh(m))], \
            (f"{month_zh(m)}，并与前一年同一个月对比（只写{month_zh(m)}和“去年同月/前一年同期”之类的说法，"
             f"不要写出前一年的年份数字）")
    options = []
    for season in SEASONS:
        for year in sorted({m.year for m in months}):
            ms = season_months(season, hemisphere, year)
            if all(x in available for x in ms):
                options.append((season, year, ms))
    season, year, ms = rng.choice(options)
    phrase = season_phrase(season, hemisphere, year)
    side = "北" if hemisphere == "N" else "南"
    hint = (f"该海域{phrase}（当地季节，海域在{side}半球）。请求里只说季节（如“{phrase}”），不要写出具体月份，"
            f"也不要说明季节如何换算成月份")
    if kind == "season_monthly":
        return [group([x], month_zh(pd.Period(x, "M"))) for x in ms], hint + "；要求季节内每个月分别统计"
    return [group(ms, f"{phrase}（{'、'.join(str(int(x[-2:])) for x in ms)}月）")], hint + "；整个季节作为一个整体统计"


def analysis_facts(spec):
    facts = []
    if "mld" in spec["analyses"]:
        facts.append("温度阈值法混合层深度：各时段的可估计剖面数、中位数和四分位数，以及无法给出精确深度的剖面数和原因"
                     "（缺测、深度不够、质量不合格等）")
    if "consistency" in spec["analyses"]:
        c = spec["consistency"]
        top, bottom = c["depth_range_m"]
        depth = (f"深度从最浅的0.49米层一直到{bottom:g}米（必须明确提到0.49米或“最表层0.49米”）" if c["shallow_requested"]
                 else f"深度为上层{bottom:g}米（用“上层/以浅/前{bottom:g}米”之类的说法，不要提最表层或0.49米）")
        facts.append(f"{VARIABLE_ZH[c['variable']]}的观测（EN4）与再分析（GLORYS）一致性：差值方向是再分析减观测"
                     f"（要么照此表述，要么不提方向，绝不能写成观测减再分析），逐个标准层给出 "
                     f"bias、MAE、RMSE 和有效配对数；{depth}")
    return facts


def new_spec(split, family, block, tier, groups, period_fact, kind, request_type, rng):
    box = block_box(block)
    spec = {"split": split, "family": family, "request_type": request_type, "kind": kind,
            "density_tertile": ["low", "mid", "high"][tier], "block": list(block),
            "region": [list(box[0]), list(box[1])], "groups": groups, "period_fact": period_fact, "analyses": [],
            "method_cards": [], "qc_policy": QC_POLICY, "dataset_revision": REVISION}
    if family in ("mld", "report"):
        spec["analyses"].append("mld")
        spec["method_cards"].append("mld_temperature_v1")
    if family != "mld":
        variable = "salinity" if family == "consistency_salinity" else "potential_temperature"
        shallow = rng.random() < .25
        spec["analyses"].append("consistency")
        spec["method_cards"].append("profile_consistency_v1")
        spec["consistency"] = {"variable": variable, "shallow_requested": shallow,
                               "depth_range_m": [.49 if shallow else 5., float(rng.choice(DEPTHS))]}
    return spec


def followups_for(spec, blocks):
    options = []
    if "consistency" in spec["analyses"]:
        options += ["switch_variable", "change_depth"]
        if "mld" not in spec["analyses"]:
            options.append("add_mld")
    if any(len(g["months"]) > 1 for g in spec["groups"]) or len(spec["groups"]) > 1:
        options.append("narrow_month")
    if len(spec["groups"]) == 1 and len(spec["groups"][0]["months"]) == 3:
        options.append("split_monthly")
    lat0, lon0 = spec["block"]
    if any(b in blocks for b in ((lat0 + 10, lon0), (lat0 - 10, lon0), (lat0, lon0 + 10), (lat0, lon0 - 10))):
        options.append("shift_region")
    return options


def apply_followup(spec, kind, rng, blocks):
    """-> (final spec, writer-facing description of the change)."""
    final = json.loads(json.dumps({k: v for k, v in spec.items() if k not in ("period_fact",)}))
    if kind == "switch_variable":
        c = final["consistency"]
        new = "salinity" if c["variable"] == "potential_temperature" else "potential_temperature"
        c["variable"] = new
        if final["family"] != "report":
            final["family"] = "consistency_salinity" if new == "salinity" else "consistency_temperature"
        elif new == "salinity":
            final["family"] = "report"
        return final, f"把一致性分析的变量从{VARIABLE_ZH[spec['consistency']['variable']]}换成{VARIABLE_ZH[new]}，其他都不变"
    if kind == "change_depth":
        c = final["consistency"]
        top, bottom = c["depth_range_m"]
        if rng.random() < .5:
            c["shallow_requested"] = not c["shallow_requested"]
            c["depth_range_m"][0] = .49 if c["shallow_requested"] else 5.
            what = (f"深度改为从最浅的0.49米层开始（到{bottom:g}米不变，必须明确提到0.49米或最表层）" if c["shallow_requested"]
                    else f"深度改为上层{bottom:g}米、不需要最表层那几层（不要再提0.49米）")
        else:
            c["depth_range_m"][1] = float(rng.choice([d for d in DEPTHS if d != bottom]))
            what = f"深度下限改为{c['depth_range_m'][1]:g}米，起始深度的要求不变"
        return final, what + "，其他都不变"
    if kind == "add_mld":
        final["analyses"] = ["mld", "consistency"]
        final["method_cards"] = ["mld_temperature_v1", "profile_consistency_v1"]
        final["family"] = "report"
        return final, "再加上同一区域、同样时段划分的温度阈值混合层深度统计，一致性分析也保留"
    if kind == "narrow_month":
        months = [m for g in spec["groups"] for m in g["months"]]
        m = rng.choice(months)
        final["groups"] = [group([m], month_zh(pd.Period(m, "M")))]
        return final, f"只看其中的{month_zh(pd.Period(m, 'M'))}这一个月，其他要求不变"
    if kind == "split_monthly":
        final["groups"] = [group([m], month_zh(pd.Period(m, "M"))) for m in spec["groups"][0]["months"]]
        return final, "把这个时段拆开，每个月分别统计，其他要求不变"
    lat0, lon0 = spec["block"]
    moves = [(b, d) for b, d in (((lat0 + 10, lon0), "北"), ((lat0 - 10, lon0), "南"), ((lat0, lon0 + 10), "东"),
                                 ((lat0, lon0 - 10), "西")) if b in blocks]
    block, direction = rng.choice(moves)
    box = block_box(block)
    final["block"], final["region"] = list(block), [list(box[0]), list(box[1])]
    return final, f"区域整体往{direction}平移10度（不要写出新的边界数字，只说平移方向和距离），其他要求不变"


def sample_specs(split, count, data, seed=0):
    """Predeclared sampling of v2 specs for one split (no text yet)."""
    rng = random.Random(seed)
    months = split_months(split)
    blocks = dev_blocks(data.index, split)
    ordered = sorted(blocks, key=lambda b: blocks[b])
    tertiles = np.array_split(np.array(ordered, dtype=object), 3)
    plan = []
    for rtype, share in TYPE_SHARE.items():
        for family, fshare in FAMILY_SHARE.items():
            plan += [(rtype, family)] * round(count * share * fshare)
    while len(plan) < count:
        plan.append(("paraphrase", "mld"))
    plan = plan[:count]
    specs, seen = [], set()
    for i, (rtype, family) in enumerate(plan):
        tier = i % 3
        while True:
            block = tuple(int(x) for x in rng.choice(list(tertiles[tier])))
            hemisphere = "N" if block[0] >= 0 else "S"
            if rtype == "reasoning":
                kind = rng.choice(REASONING_KINDS[family])
                groups, fact = reasoning_groups(rng, kind, months, hemisphere)
            else:
                kind = rng.choice(V1_KINDS[family])
                groups, fact = v1_groups(rng, kind, months)
            key = (rtype, family, block, kind, tuple(tuple(g["months"]) for g in groups))
            if key not in seen:
                seen.add(key)
                break
        spec = new_spec(split, family, block, tier, groups, fact, kind, rtype, rng)
        if rtype == "multiturn":
            follow = rng.choice(followups_for(spec, blocks))
            final, change = apply_followup(spec, follow, rng, blocks)
            final["request_type"], final["followup"] = "multiturn", {"type": follow, "change": change}
            final["first"] = spec
            spec = final
        spec["id"] = f"{split}2-{i:04d}"
        specs.append(spec)
    return specs


def writer_messages(spec, attempt, seed=0, followup=False):
    """Prompt for the teacher to write one user message (the first request, or the follow-up of a multiturn task)."""
    rng = random.Random(f"{seed}:{spec['id']}:{attempt}:{followup}")
    persona, tone, coords = rng.choice(PERSONAS), rng.choice(TONES), rng.choice(COORDINATE_STYLES)
    if followup:
        system = ("你在为一个海洋数据分析助手构造真实的多轮对话。用户刚才提了一个请求，助手已经完成并交付了结果。"
                  "现在请以同一位用户的口吻写一条简短的追问，只表达下面这个改动，自然口语即可，不要重复整段原始请求，"
                  "不要增加任何别的分析要求。只输出这条追问本身。")
        user = f"用户身份：{persona}\n原始请求：{spec['text']}\n要表达的改动：{spec['followup']['change']}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}], \
            {"persona": persona, "tone": None, "coordinates": None}
    base = spec.get("first", spec)
    facts = [f"区域（原样使用这段写法，数字和方位字不能改）：{region_text(base['region'], coords)}",
             f"时段：{base['period_fact']}", *[f"分析：{x}" for x in analysis_facts(base)]]
    system = ("你在为一个海洋数据分析助手构造真实用户的请求。请扮演给定身份、按给定风格，用自然的中文写一条发给助手的请求消息。"
              "必须准确传达“必须包含的内容”里的每一项，数字要准确；不能增加任何其他分析要求（其他区域、变量、时段、方法、"
              "敏感性对比等）。年份要写明确，不要用“去年”“今年”这类没有基准的说法（除非要求里明确要你用“去年同月”）。"
              "可以加一两句真实的背景或用途，但不要编造数据结论；可以不提海域名称，如果提，必须确实位于该坐标范围内。"
              "不要照搬这些模板化句式："
              + "；".join(V1_PHRASES) + "。只输出这条请求消息本身，不要加引号或说明。")
    user = (f"用户身份：{persona}\n风格：{tone}\n必须包含的内容：\n- " + "\n- ".join(facts))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], \
        {"persona": persona, "tone": tone, "coordinates": coords}


def extractor_messages(texts):
    """Independent re-extraction of the requirements from user text only (no spec)."""
    system = ("你是一个严格的解析器，把用户给海洋剖面分析助手的请求解析成结构化参数。只依据用户文字，按下面的约定解析，"
              "不要猜测文字里没有的内容。\n" + CONVENTIONS + "\n如果有多条用户消息，后面的是追问：按追问之后的完整最新要求"
              "解析（追问没改的部分沿用前文）。\n只输出 JSON 对象："
              '{"lat_min":数,"lat_max":数,"lon_min":数,"lon_max":数,"groups":[["YYYY-MM",...],...],'
              '"analyses":[...],"variable":字符串或null,"depth_min":数或null,"depth_max":数或null,'
              '"direction":字符串或null,"extra":"..."}')
    body = "\n".join(f"用户消息{i + 1}：{t}" for i, t in enumerate(texts))
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


def mismatches(spec, extracted):
    """Fields where an extraction differs from the spec (empty list = the text conveys exactly the spec)."""
    try:
        e = json.loads(extracted) if isinstance(extracted, str) else extracted
        (la0, la1), (lo0, lo1) = spec["region"]
        bad = []
        if any(abs(float(e[k]) - v) > 1e-6 for k, v in (("lat_min", la0), ("lat_max", la1), ("lon_min", lo0), ("lon_max", lo1))):
            bad.append("region")
        want = sorted(tuple(g["months"]) for g in spec["groups"])
        got = sorted(tuple(sorted(g)) for g in e["groups"])
        if got != want:
            bad.append("groups")
        if sorted(e["analyses"]) != sorted(spec["analyses"]):
            bad.append("analyses")
        c = spec.get("consistency") if "consistency" in spec["analyses"] else None
        if (e.get("variable") or None) != (c["variable"] if c else None):
            bad.append("variable")
        if c:
            top, bottom = c["depth_range_m"]
            if e.get("depth_min") is None or e.get("depth_max") is None or abs(float(e["depth_min"]) - top) > 1e-6 \
                    or abs(float(e["depth_max"]) - bottom) > 1e-6:
                bad.append("depth")
        if e.get("direction") == "observation_minus_reanalysis":
            bad.append("direction")
        if str(e.get("extra") or "").strip():
            bad.append("extra")
        return bad
    except (ValueError, KeyError, TypeError) as exc:
        return [f"unparsable: {str(exc)[:60]}"]
