✨ Store a modelled socket through its own model

## Problem

A task whose input is a `Decimal` could not be submitted at all. Sockets serialize one leaf at a time, `general_serializer` has no entry point for the type, and the submission failed at submit time with a serialization error — loud, but with nowhere to go, because nothing in the stack knew how to write the value and nothing was allowed to say so.

```python
class MoneyInputs(BaseModel):
    amount: Decimal

    @field_serializer("amount")
    def _dump_amount(self, value: Decimal) -> str:
        return str(value)

@task(input_model=MoneyInputs)
def double_money(amount):
    return {"kind": type(amount).__name__, "doubled": str(amount * 2)}
```

The model already says how to write the value, and now the adapter asks it. `amount` is stored as `'0.10'`, the string the model rendered, and the body is handed a `Decimal` back — so `str(amount * 2)` is exactly `'0.20'`, never a float's `'0.2000000000000000041...'`. An identical task declared with the same sockets and no model is the control: it still cannot store the value at all.

This is the aiida-workgraph half of `input_model=`/`output_model=`. The contract itself — which sockets a model declares, and the three moments its rules are held to — is node-graph's, on its own `input-model` branch (scinode/node-graph#\<PR\>), and its tests live there. What is here is the adapter concern and the decorator plumbing.

## Changes

**The model renders its own leaves on the way to storage.** `AiidaSerializationAdapter.serialize` asks node-graph's `model_dumper_for_socket` whether a task's input model owns the socket; if one does, the value goes through the model instead of through `_flatten_enums`. The walk resolves a socket's whole path through the model tree, crossing nested models and typed mappings alike, so a `Decimal` three levels down under `items['a'].inner.value` is rendered by the `Amount` model that declares it, not by whatever sits above. A socket the walk cannot place — below an unparameterized `dict` leaf, say — still falls through to the generic serializer and still fails as loudly as before.

**`Enum` handling falls out rather than being built.** An `Enum` field is stored as its bare value and arrives in the body as the member, from `model_dump(mode='json')` and `model_validate` with no enum-specific code and no dotted class path written into the task. `_flatten_enums` — this package's workaround for the same gap — is simply not reached on a task with a model.

**The decorators accept the keywords and build in this package's vocabulary.** `@task(input_model=M)`, `@task(output_model=M)` and `@task.graph(input_model=M)` call node-graph's `apply_models` with aiida-workgraph's `SocketSpecAPI`, so a modelled `int` field becomes a `workgraph.int` socket rather than node-graph's. `spec_from_model` is re-exported from `socket_spec` bound to the same vocabulary, for a task that wants a model's sockets without its rules.

**A graph task's contract is held where this engine expands it.** node-graph's checkpoint sits in `materialize_graph`, which `GraphTask.execute` calls, so a `@task.graph(input_model=M)` inside a submitted process is checked against `M` at the moment its inputs are known — including a subgraph whose bound an upstream task produced, which no earlier moment could have seen.

**Nothing changes for a task without a model.** The decorators take the same path, the spec is the same spec, and the serializer takes the branch it took before.

## Testing

`tests/test_input_model.py`, 31 tests, all against a live profile — this file asserts only what needs the engine; the contract's own tests are node-graph's.

- **Storage claims are paired with a control task** that has the same sockets and no model, so the model is visibly what did the work: the enum control's body reports `str` where the modelled body reports `Color`, and the `Decimal` control cannot be stored at all.
- **The deep-leaf walk is parameterized** over a nested model, a mapping of models, and a mapping of models with a model inside; each asserts both the stored string and the type the body received, which is what distinguishes "the model rendered it" from "something upstream happened to cope".
- **Rules that reach the run edge are read off the process**, not off an exception: exit status 323 with a message naming the task and the model, so the failure is one a user would actually see in `verdi process report`.
- **The graph contract is tested in two shapes**: a bad graph fails before it becomes a process at all (`node.process is None`, task `FAILED`, parent non-zero), and a subgraph whose bound comes from an upstream task fails once that task has run.
- **Full suite: 21 failed, 236 passed, 6 skipped.** The same 21 failures, by name, on the base commit with the same node-graph checkout (205 passed there; the 31 extra passes are this file). Nineteen are `AttributeError: 'TaskHandle' object has no attribute 'build'` — this package's `TaskHandle` against node-graph's `GraphTaskHandle` split — and are the known pairing noise, untouched by this branch.
