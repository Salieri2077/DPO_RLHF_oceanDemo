import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer
from agent.ocean import prompt_ids, read_jsonl
from agent.react import ReactTools, ScenarioTools, run as local_run
HAS_SDK = importlib.util.find_spec('agents') is not None
if HAS_SDK:
    from agent.sdk import run

ROOT = Path(__file__).resolve().parents[1]
CALL = '<tool_call>{"name":"marine_calculate","arguments":{"operation":"distance","speed":12,"hours":3}}</tool_call>'


@unittest.skipUnless(HAS_SDK, 'optional SDK environment required')
class SDKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(ROOT / "model")

    def generator(self, texts):
        sequence = iter(texts)
        def generate(ids, limit):
            tokens = self.tok.encode(next(sequence), add_special_tokens=False) + [self.tok.eos_token_id]
            self.assertLessEqual(len(tokens), limit)
            return tokens, [-1.] * len(tokens)
        return generate

    def test_sdk_owns_loop_and_preserves_prefixes(self):
        text = ['<plan>计算航程。</plan>' + CALL, '<final>36 km</final>']
        with patch('agent.react.run', side_effect=AssertionError('Must use SDK Runner')), \
             patch('socket.socket.connect', side_effect=AssertionError('Local runtime must not connect to network')):
            sdk = run('求航程', self.tok, ReactTools([]), self.generator(text))
        old = local_run('求航程', self.tok, ReactTools([]), self.generator(text))
        self.assertEqual(sdk['final'], '36 km')
        self.assertEqual(sdk['calls'], old['calls'])
        self.assertEqual(sdk['messages'], old['messages'])
        for a, b in zip(sdk['rounds'], old['rounds']):
            for key in ('input_ids', 'completion_ids', 'old_logps'):
                self.assertEqual(a[key], b[key])
        self.assertEqual(sdk['rounds'][1]['input_ids'], prompt_ids(self.tok, sdk['messages'][:-1]))

    def test_errors_budgets_and_retry(self):
        for bad in ('<tool_call>{}', '<tool_call>{"name":"shell","arguments":{}}</tool_call>',
                    CALL.replace('12', 'NaN'), CALL.replace('"distance"', '[]')):
            t = run('test', self.tok, ReactTools([]), self.generator([bad, CALL, '<final>36 km</final>']))
            self.assertIn('error', t['calls'][0]['result'])
            self.assertEqual(t['calls'][1]['result']['value'], 36)
        t = run('test', self.tok, ScenarioTools(ReactTools([]), transient=True), self.generator([CALL, CALL, '36 km']))
        self.assertEqual(t['calls'][1]['result']['value'], 36)
        for texts, kwargs, stop in (([CALL]*3, {}, 'repeated_action'),
                                    ([CALL], {'max_turns': 1}, 'turn_limit'),
                                    ([CALL]*2, {'max_calls': 1}, 'call_limit'),
                                    ([], {'max_total_len': 1}, 'context_limit')):
            t = run('test', self.tok, ReactTools([]), self.generator(texts), **kwargs)
            self.assertEqual(t['stop'], stop)
        t = run('test', self.tok, ReactTools([]), lambda ids, limit: ([42], []))
        self.assertEqual(t['stop'], 'generation_limit')

    def test_clarify_and_history(self):
        history = [{'role': 'assistant', 'content': '<clarify>速度是多少？</clarify>'},
                   {'role': 'user', 'content': '12 km/h，3 h'}]
        t = run('求航程', self.tok, ReactTools([]), self.generator([CALL, '<final>36 km</final>']), history=history)
        self.assertEqual(t['messages'][2:4], history)
        t = run('求航程', self.tok, ReactTools([]), self.generator(['<clarify>速度是多少？</clarify>']))
        self.assertEqual(t['stop'], 'clarify')

    def test_six_task_types_use_existing_verifier(self):
        from agent.sdk import run_trajectory
        from scripts.prepare_ocean_react import oracle
        data = ROOT / 'data/processed/react-v2'
        tasks = read_jsonl(data / 'ocean_agent_train.jsonl')
        env = ReactTools(read_jsonl(data / 'ocean_agent_corpus_train.jsonl'))
        for kind in ('calculate', 'retrieve', 'chain', 'recovery', 'clarify', 'direct'):
            task = next(t for t in tasks if t['kind'] == kind)
            trace = run_trajectory(task, self.tok, env, self.generator(oracle(task)))
            self.assertEqual(trace['metrics']['success'], 1, trace)


if __name__ == '__main__':
    unittest.main()
