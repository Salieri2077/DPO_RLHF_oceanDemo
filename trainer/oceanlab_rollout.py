"""Many SDK episodes at once, their model calls answered by batched generation (agent RL rollouts and validation).

Each episode runs the ordinary SDK harness (`agent.sdk.run_trajectory`) in its own thread, with a `generate(ids, limit)`
that only queues the request. Once every unfinished episode is waiting for the model, the queued prompts are generated
together, so a round of N episodes costs about one decoding pass instead of N. Tool execution and scoring stay in the
episode threads (about 1% of an episode's time; generation is the rest).

`hf_batch_generator` left-pads the prompts and splits them into sub-batches whose rows x (longest prompt + new-token
limit) stay under a token budget: on a 2080 Ti (11 GiB) the KV cache, not compute, limits the batch.
"""
import threading

import torch


class BatchedEpisodes:
    """Runs jobs concurrently; `generate_batch([(job index, ids, limit), ...]) -> [tokens, ...]` serves their calls."""

    def __init__(self, generate_batch):
        self.generate_batch = generate_batch
        self.batch_sizes = []  # requests answered per batched call, for throughput logging

    def run(self, jobs):
        """jobs: callables job(generate) -> result, where generate(ids, limit) -> (tokens, []) as for the SDK adapter.
        Returns the results in job order; re-raises the first error after every episode has ended."""
        condition = threading.Condition()
        pending, results, errors = [], [None] * len(jobs), [None] * len(jobs)
        alive = [len(jobs)]

        def generator(index):
            def generate(ids, limit):
                slot = {}
                with condition:
                    pending.append((index, list(ids), limit, slot))
                    condition.notify_all()
                    while "tokens" not in slot and "error" not in slot:
                        condition.wait()
                if "error" in slot:
                    raise RuntimeError("batched generation failed") from slot["error"]
                return slot["tokens"], []
            return generate

        def worker(index):
            try:
                results[index] = jobs[index](generator(index))
            except BaseException as exc:  # reported after all episodes end
                errors[index] = exc
            finally:
                with condition:
                    alive[0] -= 1
                    condition.notify_all()

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(jobs))]
        for thread in threads:
            thread.start()
        while True:
            with condition:
                while alive[0] and len(pending) < alive[0]:
                    condition.wait()
                if not alive[0]:
                    break
                batch = pending[:]
                pending.clear()
            try:
                outputs = self.generate_batch([(index, ids, limit) for index, ids, limit, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("generate_batch returned a different number of outputs")
                failure = None
            except BaseException as exc:
                outputs, failure = [None] * len(batch), exc
            self.batch_sizes.append(len(batch))
            with condition:
                for (_, _, _, slot), tokens in zip(batch, outputs):
                    if failure:
                        slot["error"] = failure
                    else:
                        slot["tokens"] = tokens
                condition.notify_all()
        for thread in threads:
            thread.join()
        for error in errors:
            if error is not None:
                raise error
        return results


def hf_batch_generator(model, tokenizer, device, sample, token_budget=32768, temperature=1.0):
    """Batched `model.generate` for BatchedEpisodes. Each row stops at its first end-of-turn token (kept, as the
    single-sequence generator does) or at its own limit; a row without the end token is a generation-limit stop."""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    @torch.no_grad()
    def run(rows):
        longest = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), longest), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), longest), dtype=torch.long)
        for row, (ids, _) in enumerate(rows):
            input_ids[row, longest - len(ids):] = torch.tensor(ids)
            mask[row, longest - len(ids):] = 1
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                                    max_new_tokens=max(limit for _, limit in rows), do_sample=sample,
                                    temperature=temperature if sample else None, top_p=1. if sample else None,
                                    top_k=0 if sample else None, eos_token_id=eos, pad_token_id=pad, use_cache=True)
        out = []
        for row, (_, limit) in enumerate(rows):
            tokens = output[row, longest:].tolist()[:limit]
            out.append(tokens[:tokens.index(eos) + 1] if eos in tokens else tokens)
        return out

    def generate_batch(requests):
        order = sorted(range(len(requests)), key=lambda i: len(requests[i][1]))
        outputs, group = [None] * len(requests), []

        def flush():
            for i, tokens in zip(group, run([requests[i][1:] for i in group])):
                outputs[i] = tokens

        for i in order:
            width = max(len(requests[j][1]) + requests[j][2] for j in group + [i])
            if group and (len(group) + 1) * width > token_budget:
                flush()
                group = []
            group.append(i)
        flush()
        return outputs

    return generate_batch
