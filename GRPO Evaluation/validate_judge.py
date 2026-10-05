"""Live acceptance gate; saves raw scores and exits nonzero on any failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from evaluator import ROOT, atomic_json
from trainer.trainer_utils import OCEAN_JUDGE_PROMPT, SiliconFlowRewardModel


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/judge32b_gate.json')
    parser.add_argument('--model', default='Qwen/Qwen3-32B')
    args = parser.parse_args()
    judge = SiliconFlowRewardModel(os.environ['SILICONFLOW_API_KEY'], args.model)
    source = ROOT / 'GRPO Evaluation/results/moe/generations_and_scores.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    cases = []
    for index in (0, 1, 20, 44):
        row = rows[index]
        cases.append((f'real_near_duplicate_{index}', row['question'], row['reference'],
                      [row['sft_answer'], row['grpo_answer']], 'bad_similar'))
    q = '海水为什么是咸的？'
    ref = '海水含有溶解盐类，河流输送岩石风化产生的离子，海底活动也贡献离子；蒸发移走水但留下盐。'
    cases.extend([
        ('identical_good', q, ref, [ref, ref], 'identical'),
        ('good_bad_group4', q, ref, [ref, '盐来自岩石风化等来源，河流把离子带入海洋，蒸发使盐留在海中。',
                                   '海水是咸的。' * 30, '因为鲸鱼每天往海里撒食盐。'], 'ranking'),
    ])
    records = []
    for name, question, reference, answers, kind in cases:
        for repeat in range(2):
            forward = judge.score_group(question, reference, answers)
            reverse = list(reversed(judge.score_group(question, reference, list(reversed(answers)))))
            checks = {'order_consistency': max(abs(a-b) for a,b in zip(forward, reverse)) <= 0.5}
            for label, scores in [('forward', forward), ('reverse', reverse)]:
                if kind == 'identical':
                    checks[label] = scores[0] == scores[1] and min(scores) >= 2
                elif kind == 'bad_similar':
                    checks[label] = abs(scores[0]-scores[1]) <= 0.5 and max(scores) <= -2
                else:
                    checks[label] = min(scores[:2]) >= 2 and max(scores[2:]) <= -2
            records.append({'case': name, 'repeat': repeat, 'question': question,
                            'reference': reference, 'answers': answers,
                            'forward': forward, 'reverse_aligned': reverse, 'checks': checks})
            passed = all(all(r['checks'].values()) for r in records)
            atomic_json(args.output, {'model': judge.model, 'prompt': OCEAN_JUDGE_PROMPT,
                'prompt_sha256': hashlib.sha256(OCEAN_JUDGE_PROMPT.encode()).hexdigest(),
                'complete': len(records) == len(cases)*2, 'passed': passed, 'records': records})
            print(name, repeat, forward, reverse, checks, flush=True)
    if not passed:
        raise SystemExit('Judge acceptance failed; do not launch training.')
    print('PASS: all live judge checks passed', flush=True)


if __name__ == '__main__':
    main()
