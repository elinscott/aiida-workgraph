# `input_model=` / `output_model=`: a Pydantic model as a task's wire contract

`@task(input_model=M)` makes `M` the task's input contract. `M` declares the sockets and their defaults, `M`'s serialization decides what is stored, and `M`'s rules are held to at three moments between the line that wires the task and the line that runs it. The body keeps a plain-Python signature and receives plain Python: for an `Enum` field the true member, for a `Decimal` field a `Decimal`. `@task(output_model=M)` is the mirror on the way out: `M` declares the output sockets and validates the return value at task exit, so a missing or mistyped output fails at the task that produced it rather than at the task that consumed it.

The feature belongs to node-graph, which owns sockets, specs and graph expansion. aiida-workgraph adds one thing: when a task's model owns a socket, the model renders that socket's stored value, so a field type JSON cannot hold reaches the database through the `field_serializer` that declares its form.

## Who owns what

| Concern | Home |
|---|---|
| `input_model=` / `output_model=` on `@task` and `@task.graph` | node-graph `decorator.py`, aiida-workgraph `decorator.py` (its own decorators, same call into node-graph) |
| The spec a model describes, the contract check, all three checkpoints | node-graph `input_model.py` |
| The socket vocabulary a spec is built in | each package's `SocketSpecAPI`, passed to `spec_from_model` |
| Storage: rendering a socket's value through the model that declares it | aiida-workgraph `serialization.py`, calling node-graph's `model_dumper_for_socket` |
| Reading the Python behind a storage node, for the graph checkpoint | aiida-workgraph `serialization.py`'s `to_python` |

## Definition time

`spec_from_model` calls `SocketSpecAPI.from_model` and then overlays what the model knows and `from_model` does not:

- **requiredness**, which `from_model` leaves at `True` whatever the field's default, so a field defaulting to `None` would otherwise read as a missing required input;
- **defaults rendered JSON-safe**, because the spec is persisted with the task and AiiDA refuses to store a non-JSON value in node attributes;
- **`dict[str, T]` as a typed dynamic namespace**, so each key of a runtime-sized mapping is a socket of its own, shaped by `T`.

The `structured_type` descriptor is stripped: it names a class by dotted import path, and under a model contract nothing needs it — the model is reached through the task's executor rather than through a path recorded in the data.

Model and signature are checked against each other, and any disagreement raises `ModelContractError` naming the offender: a field no parameter names, a parameter no field declares, an annotation that contradicts its field, `*args`/`**kwargs`, and a default written in the signature ("defaults live in the model — move it"). Declaring the same sockets twice (`input_model=` together with `inputs=`) is refused as well.

Two model shapes are refused rather than guessed at. `extra='allow'` admits fields nothing declares, which is the one shape a contract cannot describe; a mapping keyed by anything but `str` cannot become socket names. A bare unparameterized `dict` names no member type and stays a leaf blob, which is the honest reading of it.

## Checkpoint A — the value written at the call

Hook: `BaseHandle.__call__`, between `_prepare_call_inputs` and `task.set_inputs`.

Every value written at a task call that is *not* a socket reference is checked against the field it is written to. A reference passes untouched: it stands for a value nobody has yet.

The mechanism is a **flat shadow model**, generated once per model with `create_model` and no `__base__`, every field re-annotated `Annotated[T, WrapValidator(...)]` recursively through containers, nested models, `Optional` and union arguments. The wrap returns a socket reference or a tagged value unchanged and calls the inner validator on anything else. Wrapping only the outermost level is not enough — a reference written into one member of a `dict[str, int]` would reach the `int` validator, which refuses it.

Two properties are load-bearing:

- **Validate and discard.** The instance is thrown away and the original kwargs are passed on unchanged. Pydantic strips the proxy a tagged value wears for `int`, `float`, `bool`, `list`, `dict`, `tuple`, `Optional`, unions, `Literal`, `Enum` and nested models alike — it survives only under `Any` and `Path` — and a stripped value is a literal, so forwarding the validated copy would silently turn a link into a copy of the graph input's current value.
- **No base class.** Inheriting the user's model would run their `@field_validator`/`@model_validator` here, where a tagged value forwards comparisons and arithmetic, so a rule written for resolved inputs would be judged against a placeholder and fail — or pass — silently.

The cost is the documented limit: **a user's field and model validators do not fire at A**. A field validator capping `a <= 100` lets `a=1000` through at wiring; checkpoints B and C catch it. For the same reason a `mode='before'` normalizer is not honoured at A: the shadow carries types only, so a field that means to accept a flat list and store rows needs a widened annotation (`list[int] | list[list[int]]`) or belongs on the outermost model.

What A actually adds over the socket layer's own type check is narrower than it looks and worth stating plainly: a leaf socket whose identifier is `int`, `float` or an enum already refuses a bad literal at `set_inputs`. A catches the fields the type map reads as `any` or `annotated` — `Decimal`, `tuple[int, int]`, `Field(gt=0)` and anything else expressed as a constraint rather than a type — and gives every other case one error class and the model's own field path.

## Checkpoint B — a graph's resolved inputs

Hook: `materialize_graph`, after `coerce_inputs_from_spec` and `_deserialize_inputs`, before the body runs.

A `@task.graph`'s inputs are values by the time its body runs, so the **real** model runs here, cross-field rules and all. An untagged *copy* is validated (`untagged_copy`, not the in-place `resolve_tagged_values`) and discarded; the body receives the original tagged values, because it turns those tags into links and a fresh object carries none.

All three expansion paths funnel through `materialize_graph` — `GraphTaskHandle.build`, `BaseEngine._build_subgraph`, and aiida-workgraph's `GraphTask.execute` — so one hook covers a graph built in a session, a subgraph built by the local engine, and a graph task expanded inside a submitted process. A nested graph fed a value another task produced is checked at its own boundary, which is the first moment that value exists.

A graph body is handed whatever the engine wraps its values in — under AiiDA, storage nodes — because that is what it needs to draw links. A model declaring `str` cannot be asked to accept an `orm.Str`, so the checkpoint asks the graph's serialization adapter for the plain Python behind each value first (`SerializationAdapter.to_python`, the identity for an adapter that wraps nothing; aiida-workgraph's renders each node through `aiida-pythonjob`'s deserializers and leaves a node no deserializer can render as it is).

`output_model=` is refused on `@task.graph`: a graph body returns socket references, so there is nothing to validate against the model.

## Checkpoint C — the leaf run edge

The executor is a wrapper around the body. Inputs are validated once the engine has assembled and deserialized them — for a function task, inside `aiida-pythonjob`'s `PyFunction.run`, after the AiiDA nodes are back to raw Python and after `coerce_inputs_from_spec` — and the body receives `dict(validated)`: the model's own field values as Python objects, so an `Enum` field arrives as the member and a `Decimal` as a `Decimal`. A failure is a `TaskInputValidationError` before the body executes, surfacing as exit status 323 with a message naming the function and the model.

The spec is built from the *undecorated* function, so its signature, source and return annotation stay visible to inference; only the executor is swapped.

## Validation may change representation, never content

At B and C, a model is allowed to change how a value is spelled and not what it says. `'60'` may become a `Decimal`, `'none'` an `Enum` member, a list a tuple — all the same content. Deriving or rewriting a value is refused: the body would then run on a value that never reached storage, so provenance would record one input while the body saw another.

The comparison is made against a **plain twin** of the model — the same fields and constraints, built with no base class so every rule the user wrote is absent, and recursively so for nested models. The inputs as given and the values validation produced are each read through the twin and dumped `mode='json'`; a field whose two readings differ raises `ModelDerivedValueError` naming the field and the task or graph. Reading both sides through the twin is what keeps a `field_serializer` — which renders, and so is representation — out of the comparison.

Only the fields the caller supplied are compared, so a default filling an omitted field is not a change. The rule in one line: **a validator must be a no-op on values that are already resolved.** An idempotent derivation therefore passes when it is given a correct value and is refused when it rewrites one; `str.upper()` on `'SILICON'` passes and on `'silicon'` does not.

Validators written into an annotation (`Annotated[int, AfterValidator(...)]`) are part of the type: they run on both sides of the comparison and are not seen here.

## Storage — values only, rendered by their own model

`AiidaSerializationAdapter.serialize` asks `model_dumper_for_socket` whether a task's input model owns the socket; if one does, the value is rendered through the model instead of by `_flatten_enums`. `dump_model_field` places the value in `model_construct` and dumps that one field with `model_dump(mode='json', include={name})`, so a `field_serializer` declared on the model decides the stored form. The walk resolves a socket's whole path through the model tree, crossing nested models and typed mappings alike, and renders the leaf with the model that declares it; a socket the walk cannot place — under a leaf `dict`, say — falls through to the generic serializer, which stores what it knows and refuses what it does not.

Two dump modes, because storage and the body want different things. **Storage gets `model_dump(mode='json')`**, per field on the way in and whole-model on the way out, because a socket value has to survive an AiiDA node and a JSON round trip. **The body gets `dict(validated_instance)`**, so nested models stay models and an `Enum` stays a member.

The corollary is that `_flatten_enums` — the adapter's workaround for `general_serializer` having no `Enum` serializer — is dead code on an `input_model` task, short-circuited by the model branch.

## Mappings whose size is decided at runtime

`dict[str, T]` is the canonical dynamically-sized shape: the keys are unknown at definition time, the members are not, because `T` shapes every one of them. At runtime each key becomes a socket named by the key, validation runs per member, and a failure names the key it came from (`blocks.b1.width`). A downstream task whose own field is `dict[str, T]` reconstructs the members on the way in as well.

Naming *one* future member at build time does not work — the per-key sockets do not exist until the producing task has run — and that is the gap scinode/node-graph#160 closes. `list[T]` stays a leaf list socket for the same reason: a namespace addresses its members by name and a list has none to give.

node-graph deliberately does not make `dict[str, T]` dynamic outside a model contract ("no implicit dynamic unless user chose pydantic dynamic", `_child_spec_from_type`), so this is an `input_model`-only reading of the annotation. Under a contract there is no other way to spell a runtime-sized mapping, and a `model_config`-level dynamic would make the whole model open-topped rather than one field of it.

## Considered and not built

- **Type-checking a reference against the field it is written to (A2).** A socket carries an identifier, not the model's field type, and a link into a typed socket is not checked today; matching identifiers to annotations would refuse valid graphs.
- **A "skip if any input is a reference" model validator on the shadow.** It would let the user's validators run when everything is a literal, but it also skips the literal fields' own type checks in the mixed case, which is the common case.
- **Validation at assignment (`validate_assignment` on `model_construct`).** A per-field entry point that runs field validators without the whole payload; cross-field rules would still have to wait, and the two later checkpoints already cover what it would catch.
- **`list[T]` as per-member sockets.** Waiting on scinode/node-graph#160.

## Not covered

- `PythonJob`, `calcfunction`/`workfunction` and process-class tasks: only plain function tasks are wrapped, and a model on anything else is refused.
- Output storage below the top level of the output model: the return value is dumped whole by the validated instance.
- A task restored through `SafeExecutor` in pickled mode: the storage path cannot reach its model and falls back to the generic serialization path.
- A socket the model walk cannot place, below an unparameterized `dict` leaf for instance. The generic serializer takes it and fails loudly on a type it has no entry point for.
