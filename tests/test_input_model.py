"""A Pydantic model as a task's wire contract, under the AiiDA engine.

The contract itself -- which sockets a model declares, and when its rules are
held to -- is node-graph's, and its tests live there. What is asserted here is
what only this package can answer: that the model decides the *stored* form of
a value, that the body gets the rich object back out of storage, and that a
rule broken inside a submitted process fails that process with a message
naming the task.
"""

from __future__ import annotations

import enum
from decimal import Decimal

import pytest
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from aiida_workgraph import WorkGraph, task
from aiida_workgraph.socket_spec import spec_from_model
from node_graph.input_model import ModelContractError

#: aiida-pythonjob's exit status for a body that raised.
FUNCTION_FAILED = 323


class Color(enum.Enum):
    RED = 'red'
    BLUE = 'blue'


# --------------------------------------------------------------------------
# 1. The model builds this package's sockets
# --------------------------------------------------------------------------


class AddInputs(BaseModel):
    """Two summands, the second optional."""

    x: int
    y: int = 7


@task(input_model=AddInputs)
def add(x, y):
    return x + y


def test_the_sockets_carry_this_packages_identifiers():
    """The spec is built through aiida-workgraph's socket vocabulary, not node-graph's."""
    fields = add._spec.inputs.fields
    assert fields['x'].identifier == 'workgraph.int'
    assert fields['x'].meta.required is True
    assert fields['y'].identifier == 'workgraph.int'
    assert fields['y'].meta.required is False
    assert fields['y'].default == 7


def test_an_omitted_input_runs_on_the_models_default():
    wg = WorkGraph('add_default')
    node = wg.add_task(add, name='add', x=2)
    wg.run()
    assert node.outputs.result.value.value == 9


class NudgeInputs(BaseModel):
    """``by`` defaults to ``None``, the value the engine declines to store."""

    x: int
    by: int | None = None


@task(input_model=NudgeInputs)
def nudge(x, by):
    return x if by is None else x + by


def test_a_field_defaulting_to_none_is_not_a_missing_required_input():
    assert nudge._spec.inputs.fields['by'].meta.required is False
    wg = WorkGraph('nudge')
    node = wg.add_task(nudge, name='nudge', x=4)
    wg.run()
    assert node.outputs.result.value.value == 4


def test_a_model_on_a_process_task_is_refused():
    from aiida.calculations.arithmetic.add import ArithmeticAddCalculation

    with pytest.raises(ModelContractError, match='plain Python function task'):
        task(input_model=AddInputs)(ArithmeticAddCalculation)


# --------------------------------------------------------------------------
# 2. An Enum: stored as its bare value, delivered as the member
# --------------------------------------------------------------------------


class PaintInputs(BaseModel):
    color: Color


@task(input_model=PaintInputs, outputs=['is_member', 'seen'])
def paint(color):
    # ``is`` and not ``==``: the body gets the member itself, not a look-alike.
    return {'is_member': color is Color.RED, 'seen': type(color).__name__}


@task(inputs=spec_from_model(PaintInputs), outputs=['is_member', 'seen'])
def paint_without_model(color):
    """Same sockets, no contract: the control for :func:`paint`."""
    return {'is_member': color is Color.RED, 'seen': type(color).__name__}


def test_an_enum_is_stored_as_its_bare_value():
    wg = WorkGraph('paint_store')
    node = wg.add_task(paint, name='paint', color=Color.RED)
    wg.run()
    stored = dict(node.process.inputs.function_inputs)['color']
    assert stored.value == 'red'


def test_the_body_receives_the_enum_member_and_only_with_the_model():
    wg = WorkGraph('paint_body')
    with_model = wg.add_task(paint, name='with_model', color=Color.RED)
    without_model = wg.add_task(paint_without_model, name='without_model', color=Color.RED)
    wg.run()

    assert with_model.outputs.is_member.value.value is True
    assert with_model.outputs.seen.value.value == 'Color'
    # The control proves the model is what rebuilt the member: the same socket
    # hands the bare value to a body no model stands in front of.
    assert without_model.outputs.is_member.value.value is False
    assert without_model.outputs.seen.value.value == 'str'


def test_an_enum_survives_a_workgraph_round_trip():
    wg = WorkGraph('paint_round_trip')
    wg.add_task(paint, name='paint', color=Color.BLUE)
    rebuilt = WorkGraph.from_dict(wg.to_dict())
    assert rebuilt.tasks.paint.inputs.color.value == Color.BLUE
    rebuilt.run()
    assert rebuilt.tasks.paint.outputs.is_member.value.value is False
    assert rebuilt.tasks.paint.outputs.seen.value.value == 'Color'


def test_no_class_path_is_written_into_the_task():
    """The spec stored with the task names no class: the model is reached through the code."""
    assert 'structured_type' not in paint._spec.inputs.fields['color'].meta.extras


# --------------------------------------------------------------------------
# 3. A type JSON cannot hold, carried by the model's own serializer
# --------------------------------------------------------------------------


class MoneyInputs(BaseModel):
    """A ``Decimal`` amount, stored as the string the model renders."""

    amount: Decimal

    @field_serializer('amount')
    def _dump_amount(self, value: Decimal) -> str:
        return str(value)

    @field_validator('amount', mode='before')
    @classmethod
    def _load_amount(cls, value):
        return value if isinstance(value, Decimal) else Decimal(str(value))


@task(input_model=MoneyInputs, outputs=['kind', 'doubled'])
def double_money(amount):
    return {'kind': type(amount).__name__, 'doubled': str(amount * 2)}


@task(inputs=spec_from_model(MoneyInputs), outputs=['kind', 'doubled'])
def double_money_without_model(amount):
    """Same sockets, no contract: the control for :func:`double_money`."""
    return {'kind': type(amount).__name__, 'doubled': str(amount * 2)}


def test_a_field_serializer_decides_the_stored_form():
    wg = WorkGraph('money_store')
    node = wg.add_task(double_money, name='money', amount=Decimal('0.10'))
    wg.run()
    stored = dict(node.process.inputs.function_inputs)['amount']
    assert stored.value == '0.10'


def test_the_body_receives_the_decimal_and_only_with_the_model():
    wg = WorkGraph('money_body')
    node = wg.add_task(double_money, name='money', amount=Decimal('0.10'))
    wg.run()
    assert node.outputs.kind.value.value == 'Decimal'
    # Exact, because a Decimal round-tripped as a string never became a float.
    assert node.outputs.doubled.value.value == '0.20'


def test_without_the_model_the_same_value_cannot_even_be_stored():
    """The control: nothing else in the stack knows how to write a ``Decimal``."""
    wg = WorkGraph('money_control')
    wg.add_task(double_money_without_model, name='money', amount=Decimal('0.10'))
    with pytest.raises(ValueError, match='decimal.Decimal'):
        wg.run()


# --------------------------------------------------------------------------
# 4. Rules the socket layer cannot see fail the process that broke them
# --------------------------------------------------------------------------


class RangeInputs(BaseModel):
    """``low`` and ``high`` are ints the socket accepts; their order is the model's rule."""

    low: int
    high: int = Field(le=100)

    @model_validator(mode='after')
    def _ordered(self):
        if self.low >= self.high:
            raise ValueError('low must be below high')
        return self


@task(input_model=RangeInputs)
def span(low, high):
    return high - low


def test_a_cross_field_rule_fails_the_process_naming_the_task():
    wg = WorkGraph('span_bad_order')
    node = wg.add_task(span, name='span', low=9, high=3)
    wg.run()
    assert node.process.exit_status == FUNCTION_FAILED
    assert "Task 'span' got inputs RangeInputs rejects" in node.process.exit_message
    assert 'low must be below high' in node.process.exit_message


def test_a_field_constraint_fails_the_process():
    """``le=100`` is a model rule, not a socket type, so the build accepts 500."""
    wg = WorkGraph('span_too_high')
    node = wg.add_task(span, name='span', low=1, high=500)
    wg.run()
    assert node.process.exit_status == FUNCTION_FAILED
    assert 'less than or equal to 100' in node.process.exit_message


# --------------------------------------------------------------------------
# 5. A graph task's contract, held where the engine expands it
# --------------------------------------------------------------------------


class WindowInputs(BaseModel):
    """The window a graph opens; ``lower`` below ``upper`` is the graph's own rule."""

    lower: int
    upper: int

    @model_validator(mode='after')
    def _ordered(self):
        if self.lower >= self.upper:
            raise ValueError('lower must be below upper')
        return self


@task.graph(input_model=WindowInputs)
def window(lower, upper):
    return add(x=lower, y=upper)


@task()
def shrink(value):
    return value - 10


@task.graph()
def window_of_a_computed_bound(lower, upper):
    """The subgraph's bound is decided by a task, so it is a value only at run time."""
    shrunk = shrink(value=upper)
    return window(lower=lower, upper=shrunk.result)


def test_a_graph_contract_holds_when_the_graph_is_submitted():
    wg = WorkGraph('window_ok')
    node = wg.add_task(window, name='window', lower=1, upper=3)
    wg.run()
    assert node.process.exit_status == 0


def test_a_graph_contract_fails_the_submitted_graph():
    """The graph is refused as it is expanded, so it never becomes a process at all."""
    wg = WorkGraph('window_bad')
    node = wg.add_task(window, name='window', lower=9, upper=3)
    wg.run()
    assert node.process is None
    assert node.state == 'FAILED'
    assert wg.process.exit_status != 0


def test_a_runtime_value_is_checked_at_the_graph_it_reaches():
    """Nothing knows ``upper`` is 5 until ``shrink`` has run, so this is the first chance."""
    wg = WorkGraph('window_computed')
    node = wg.add_task(window_of_a_computed_bound, name='outer', lower=9, upper=15)
    wg.run()
    assert node.process.exit_status != 0


# --------------------------------------------------------------------------
# 6. Output models
# --------------------------------------------------------------------------


class SumAndProduct(BaseModel):
    sum: int
    product: int


@task(output_model=SumAndProduct)
def combine(x, y):
    return {'sum': x + y, 'product': x * y}


@task(output_model=SumAndProduct)
def combine_forgetting_product(x, y):
    return {'sum': x + y}


@task(output_model=SumAndProduct)
def combine_with_a_bad_type(x, y):
    return {'sum': 'not a number', 'product': x * y}


def test_the_output_sockets_come_from_the_output_model():
    assert set(combine._spec.outputs.fields) == {'sum', 'product'}


def test_a_return_the_model_accepts_lands_on_the_sockets():
    wg = WorkGraph('combine_ok')
    node = wg.add_task(combine, name='combine', x=2, y=3)
    wg.run()
    assert node.outputs.sum.value.value == 5
    assert node.outputs.product.value.value == 6


def test_a_missing_output_fails_at_the_source_task():
    wg = WorkGraph('combine_missing')
    node = wg.add_task(combine_forgetting_product, name='combine', x=2, y=3)
    wg.run()
    assert node.process.exit_status == FUNCTION_FAILED
    assert "Task 'combine_forgetting_product' returned outputs SumAndProduct rejects" in node.process.exit_message
    assert 'product' in node.process.exit_message


def test_a_mistyped_output_fails_at_the_source_task():
    wg = WorkGraph('combine_bad_type')
    node = wg.add_task(combine_with_a_bad_type, name='combine', x=2, y=3)
    wg.run()
    assert node.process.exit_status == FUNCTION_FAILED
    assert 'sum' in node.process.exit_message


# --------------------------------------------------------------------------
# 7. Mappings whose size is only known at runtime
# --------------------------------------------------------------------------


class Block(BaseModel):
    """One member of a mapping the task fills in at runtime."""

    width: int
    label: str


class Blocks(BaseModel):
    blocks: dict[str, Block]


class Recipe(BaseModel):
    """Two mappings on the way in: one of plain strings, one needing the model."""

    pseudos: dict[str, str]
    amounts: dict[str, Decimal] = {}

    @field_serializer('amounts')
    def _dump_amounts(self, value: dict[str, Decimal]) -> dict[str, str]:
        return {key: str(amount) for key, amount in value.items()}

    @field_validator('amounts', mode='before')
    @classmethod
    def _load_amounts(cls, value):
        return {key: amount if isinstance(amount, Decimal) else Decimal(str(amount)) for key, amount in value.items()}


@task(output_model=Blocks)
def cut_blocks(n):
    """Decide how many members the mapping has, inside the body."""
    return {'blocks': {f'b{i}': {'width': i + 1, 'label': f'block {i}'} for i in range(n)}}


@task(output_model=Blocks)
def cut_one_bad_block(n):
    return {'blocks': {'b0': {'width': 1, 'label': 'fine'}, 'b1': {'width': 'wide', 'label': 'bad'}}}


@task(input_model=Blocks, outputs=['widest'])
def widest_block(blocks):
    # Each member arrived through ``Block``, so the body reads attributes.
    return {'widest': max(item.width for item in blocks.values())}


@task(input_model=Recipe, outputs=['names', 'kinds'])
def describe_recipe(pseudos, amounts):
    return {
        'names': ','.join(sorted(pseudos)),
        'kinds': ','.join(sorted({type(amount).__name__ for amount in amounts.values()})),
    }


def test_a_typed_mapping_field_becomes_a_dynamic_namespace():
    blocks = cut_blocks._spec.outputs.fields['blocks']
    assert blocks.is_namespace()
    assert blocks.meta.dynamic is True
    # Every member is typed, so a key's sockets are known before any key is.
    assert set(blocks.item.fields) == {'width', 'label'}
    assert blocks.item.fields['width'].identifier == 'workgraph.int'


def test_a_mapping_becomes_one_socket_per_key_after_the_run():
    wg = WorkGraph('blocks_out')
    node = wg.add_task(cut_blocks, name='cut', n=3)
    wg.run()
    assert set(node.outputs.blocks._sockets) == {'b0', 'b1', 'b2'}
    assert node.outputs.blocks.b1.width.value.value == 2
    assert node.outputs.blocks.b1.label.value.value == 'block 1'


def test_a_bad_member_fails_at_the_source_task_naming_its_key():
    wg = WorkGraph('blocks_bad')
    node = wg.add_task(cut_one_bad_block, name='cut', n=2)
    wg.run()
    assert node.process.exit_status == FUNCTION_FAILED
    # The key is in the path pydantic reports, so the reader knows which member.
    assert 'blocks.b1.width' in node.process.exit_message


def test_a_downstream_task_consumes_the_mapping():
    wg = WorkGraph('blocks_chain')
    source = wg.add_task(cut_blocks, name='cut', n=3)
    consumer = wg.add_task(widest_block, name='widest', blocks=source.outputs.blocks)
    wg.run()
    assert consumer.outputs.widest.value.value == 3


def test_addressing_one_future_member_at_build_time_is_not_available_yet():
    """The per-key sockets appear when the task runs, so a build-time link cannot name one.

    Ordered and by-name member access on a socket that has not produced its
    members yet is scinode/node-graph#160.
    """
    wg = WorkGraph('blocks_future')
    source = wg.add_task(cut_blocks, name='cut', n=3)
    with pytest.raises(AttributeError, match="has no sub-socket 'b1'"):
        source.outputs.blocks.b1


def test_a_mapping_input_becomes_one_socket_per_key():
    wg = WorkGraph('recipe_in')
    node = wg.add_task(describe_recipe, name='recipe', pseudos={'Si': 'si.upf', 'O': 'o.upf'})
    assert set(node.inputs.pseudos._sockets) == {'Si', 'O'}
    wg.run()
    assert node.outputs.names.value.value == 'O,Si'


def test_a_member_of_a_mapping_is_stored_through_the_model():
    wg = WorkGraph('recipe_decimal')
    node = wg.add_task(
        describe_recipe,
        name='recipe',
        pseudos={'Si': 'si.upf'},
        amounts={'Si': Decimal('1.50'), 'O': Decimal('0.25')},
    )
    wg.run()
    stored = dict(node.process.inputs.function_inputs)['amounts']
    assert {key: value.value for key, value in dict(stored).items()} == {'Si': '1.50', 'O': '0.25'}
    assert node.outputs.kinds.value.value == 'Decimal'


class Amount(BaseModel):
    """A model used as a field, carrying a type JSON cannot hold."""

    value: Decimal

    @field_serializer('value')
    def _dump_value(self, value: Decimal) -> str:
        return str(value)

    @field_validator('value', mode='before')
    @classmethod
    def _load_value(cls, value):
        return value if isinstance(value, Decimal) else Decimal(str(value))


class Nested(BaseModel):
    cfg: Amount


class Mapped(BaseModel):
    cfgs: dict[str, Amount]


class Layer(BaseModel):
    inner: Amount


class DeepMapped(BaseModel):
    items: dict[str, Layer]


@task(input_model=Nested, outputs=['kind', 'stored'])
def read_nested(cfg):
    return {'kind': type(cfg.value).__name__, 'stored': str(cfg.value)}


@task(input_model=Mapped, outputs=['kind', 'stored'])
def read_mapped(cfgs):
    return {'kind': type(cfgs['a'].value).__name__, 'stored': str(cfgs['a'].value)}


@task(input_model=DeepMapped, outputs=['kind', 'stored'])
def read_deep(items):
    return {'kind': type(items['a'].inner.value).__name__, 'stored': str(items['a'].inner.value)}


@pytest.mark.parametrize(
    'entry_point, payload, path',
    [
        (read_nested, {'cfg': {'value': Decimal('1.50')}}, ('cfg', 'value')),
        (read_mapped, {'cfgs': {'a': {'value': Decimal('1.50')}}}, ('cfgs', 'a', 'value')),
        (read_deep, {'items': {'a': {'inner': {'value': Decimal('1.50')}}}}, ('items', 'a', 'inner', 'value')),
    ],
    ids=['nested-model', 'mapping-of-models', 'mapping-of-nested-models'],
)
def test_a_model_renders_its_own_leaf_however_deep_it_sits(entry_point, payload, path):
    """Whatever models and mappings the path crosses, the model declaring the leaf renders it."""
    wg = WorkGraph(f'depth_{path[0]}')
    node = wg.add_task(entry_point, name='read', **payload)
    wg.run()

    assert node.process.exit_status == 0
    stored = node.process.inputs.function_inputs
    for name in path:
        stored = stored[name]
    assert stored.value == '1.50'
    # And the body gets the Decimal back, not the string that was stored.
    assert node.outputs.kind.value.value == 'Decimal'
    assert node.outputs.stored.value.value == '1.50'


# --------------------------------------------------------------------------
# 8. What a task without a model does
# --------------------------------------------------------------------------


@task
def plain_add(x: int, y: int = 7) -> int:
    return x + y


def test_a_task_without_a_model_is_untouched():
    fields = plain_add._spec.inputs.fields
    assert fields['x'].identifier == 'workgraph.int'
    assert fields['y'].default == 7
    wg = WorkGraph('plain')
    node = wg.add_task(plain_add, name='add', x=2)
    wg.run()
    assert node.outputs.result.value.value == 9
