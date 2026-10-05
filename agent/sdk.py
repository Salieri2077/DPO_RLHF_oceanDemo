"""Local OceanHeart Model adapter. The OpenAI SDK Runner owns the execution loop.

No hosted models, network tracing, handoffs, or answer repair. One adapter per run.
"""
import asyncio
import copy
import json
import time
import uuid

from agents import Agent, FunctionTool, Model, ModelResponse, Runner, RunConfig, MaxTurnsExceeded
from agents.usage import Usage
from agents.exceptions import UserError
from agents.tracing import set_trace_provider
from agents.tracing.provider import DefaultTraceProvider
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from agent.ocean import TOOLS, observation, prompt_ids
from agent.react import SYSTEM, ScenarioTools, messages_for, parse_action, score

VERSION = "ocean-openai-agents-v1"
ERROR_TOOL = "_ocean_protocol_error"
# This local-only runtime deliberately disables exporters process-wide. An empty
# provider avoids even constructing the SDK's default network export client.
set_trace_provider(DefaultTraceProvider())


class BudgetStop(Exception):
    pass


class OceanModel(Model):
    def __init__(self, question, tokenizer, environment, generate, history, budget, save):
        self.tokenizer, self.environment, self.generate = tokenizer, environment, generate
        self.budget, self.save = budget, save
        self.initial = messages_for(question, history)
        self.raw = {}
        self.trace = dict(messages=copy.deepcopy(self.initial), rounds=[], calls=[], final="",
                          stop="running", unrecovered_error=False, duplicates={}, runtime=VERSION)

    def stop(self, reason):
        self.trace["stop"] = reason
        raise BudgetStop(reason)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError("OceanHeart adapter supports non-streaming Runner.run only")

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, **kwargs):
        if handoffs or output_schema or any(kwargs.get(k) for k in ("previous_response_id", "conversation_id", "prompt")):
            raise ValueError("Local adapter does not support hosted state, handoffs or output schemas")
        if system_instructions != SYSTEM:
            raise ValueError("Use the checkpoint's OceanHeart system instructions")
        # Rebuild the actual model prefix from SDK history, retaining original generated text.
        items = [{"role": "user", "content": input}] if isinstance(input, str) else input
        if not items or items[0].get("content") != self.initial[1]["content"]:
            raise ValueError("Adapter instances are single-run; unexpected initial input")
        messages = copy.deepcopy(self.initial)
        for item in items[1:]:
            if item.get("type") == "function_call":
                messages.append({"role": "assistant", "content": self.raw[item["call_id"]]})
            elif item.get("type") == "function_call_output":
                messages.append({"role": "tool", "content": item["output"]})
            else:
                raise ValueError("Unexpected SDK history item; refusing lossy prompt conversion")
        self.trace["messages"] = messages
        ids = prompt_ids(self.tokenizer, messages)
        limit = self.budget["max_new_tokens"]
        if len(ids) + limit > self.budget["max_total_len"]:
            self.stop("context_limit")
        started = time.monotonic()
        tokens, logps = self.generate(ids, limit)
        text = self.tokenizer.decode(tokens, skip_special_tokens=False).replace(self.tokenizer.eos_token, "").strip()
        entry = dict(input_ids=ids, completion_ids=tokens, old_logps=logps, text=text,
                     seconds=time.monotonic() - started)
        self.trace["rounds"].append(entry)
        if logps and len(logps) != len(tokens):
            raise ValueError("Generated token/log probability length mismatch")
        if not tokens or tokens[-1] != self.tokenizer.eos_token_id:
            self.stop("generation_limit" if tokens else "empty_generation")
        messages.append({"role": "assistant", "content": text})
        error = None
        try:
            action = parse_action(text)
            if action["type"] == "tool":
                json.dumps(action["call"], allow_nan=False)
                if action["call"]["name"] not in [t["function"]["name"] for t in TOOLS]:
                    raise ValueError("未知工具")
        except (ValueError, TypeError, RecursionError) as exc:
            error = str(exc)[:160]
            action = {"type": "invalid"}
        entry["action"] = action
        ident = uuid.uuid4().hex
        if action["type"] in {"final", "clarify"}:
            self.trace.update(final=action["content"], stop=action["type"])
            output = ResponseOutputMessage(id=ident, role="assistant", status="completed", type="message",
                content=[ResponseOutputText(type="output_text", text=action["content"], annotations=[])])
        else:
            if len(self.trace["calls"]) >= self.budget["max_calls"]:
                self.stop("call_limit")
            self.raw[ident] = text
            call = action.get("call", {})
            entry["sdk_call_id"] = ident
            output = ResponseFunctionToolCall(type="function_call", call_id=ident,
                name=ERROR_TOOL if error else call["name"],
                arguments=json.dumps({"error": error} if error else call["arguments"], ensure_ascii=False))
        return ModelResponse(output=[output], usage=Usage(requests=1, input_tokens=len(ids),
                             output_tokens=len(tokens), total_tokens=len(ids) + len(tokens)), response_id=None)

    def tool(self, name, description, schema):
        async def invoke(ctx, arguments):
            args = json.loads(arguments)
            call = {} if name == ERROR_TOOL else {"name": name, "arguments": args}
            previous = [c for c in self.trace["calls"] if c["call"] == call]
            if name == ERROR_TOOL:
                result = {"error": args["error"]}
            elif any("error" not in c["result"] for c in previous):
                key = json.dumps(call, sort_keys=True, ensure_ascii=False)
                count = self.trace["duplicates"].get(key, 0) + 1
                self.trace["duplicates"][key] = count
                result = {"error": "duplicate_successful_call", "retryable": False}
                if count >= 2:
                    self.trace["stop"] = "repeated_action"
            elif previous and (not previous[-1]["result"].get("retryable") or len(previous) >= 2):
                result = {"error": "repeated_failed_call: change arguments or clarify", "retryable": False}
            else:
                result = self.environment.execute(call)
            result = json.loads(observation(result, self.tokenizer))
            self.trace["calls"].append({"call": call, "result": result})
            self.trace["unrecovered_error"] = "error" in result
            content = json.dumps(result, ensure_ascii=False)
            self.trace["messages"].append({"role": "tool", "content": content})
            if self.save:
                self.save(self.trace)
            if self.trace["stop"] != "running":
                raise BudgetStop(self.trace["stop"])
            return content
        return FunctionTool(name=name, description=description, params_json_schema=schema,
                            on_invoke_tool=invoke, strict_json_schema=False)


def run(question, tokenizer, environment, generate, *, history=(), max_turns=6, max_calls=4,
        max_total_len=2048, max_new_tokens=192, save=None):
    """Synchronous entry for existing evaluation workers; no custom model/tool loop."""
    budget = dict(max_turns=max_turns, max_calls=max_calls, max_total_len=max_total_len, max_new_tokens=max_new_tokens)
    if min(budget.values()) < 1:
        raise ValueError("budgets must be positive")
    model = OceanModel(question, tokenizer, environment, generate, history, budget, save)
    tools = [model.tool(f["name"], f["description"], f["parameters"]) for f in (t["function"] for t in TOOLS)]
    # Internal error transport only, never included in the local model's tool prompt.
    tools.append(model.tool(ERROR_TOOL, "Protocol error feedback", {"type": "object", "properties": {"error": {"type": "string"}}}))
    agent = Agent(name="OceanHeart", instructions=SYSTEM, model=model, tools=tools)
    try:
        asyncio.run(Runner.run(agent, question, max_turns=max_turns,
                              run_config=RunConfig(tracing_disabled=True)))
    except MaxTurnsExceeded:
        model.trace["stop"] = "turn_limit"
    except BudgetStop:
        pass
    except UserError as exc:
        if not isinstance(exc.__cause__, BudgetStop):
            raise
    if model.trace["stop"] == "running":
        raise RuntimeError("SDK ended without an OceanHeart terminal action")
    if save:
        save(model.trace)
    return model.trace


def run_trajectory(task, tokenizer, environment, generate, **kwargs):
    public = task.get("public", {})
    env = ScenarioTools(environment, transient=public.get("transient", False))
    trace = run(task["question"], tokenizer, env, generate, history=public.get("history", []), **kwargs)
    trace.update(id=task["id"], kind=task["kind"])
    trace["metrics"] = score(task, trace)
    return trace
