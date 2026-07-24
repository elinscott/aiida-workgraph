"""Typed dynamic output namespaces: whole-handle and per-key entry linking.

Entries of a ``dynamic(SomeTypedDict)`` output namespace can be populated
either by assigning a whole namespace handle or key by key from leaf sockets.
Both forms must build, survive a ``to_dict``/``from_dict`` round trip (which
``run`` and the daemon perform), and propagate values at execution.
"""

from typing import Annotated, Any, TypedDict

try:
    from typing import NotRequired
except ImportError:  # pragma: no cover - Python < 3.11
    from typing_extensions import NotRequired

import pytest
from aiida import orm

from aiida_workgraph import WorkGraph, dynamic, task


class Contract(TypedDict):
    value: orm.Int
    tag: orm.Int


@task
def mk(x, k) -> orm.Int:
    return orm.Int(int(x) + int(k))


def test_superset_source_into_narrow_dynamic_entry_rejected():
    """A whole-handle source with keys beyond the entry spec must not link."""

    class SupersetOut(TypedDict):
        value: orm.Int
        tag: orm.Int
        extra: orm.Int

    @task.graph
    def sub(x: orm.Int) -> SupersetOut:
        return SupersetOut(value=mk(x, 10).result, tag=mk(x, 100).result, extra=mk(x, 1).result)

    class Out(TypedDict):
        blocks: Annotated[dict, dynamic(Contract)]

    @task.graph
    def make_blocks(a: orm.Int) -> Out:
        return Out(blocks={'b1': sub(a)})

    with pytest.raises(ValueError, match='Namespace structures do not match'):
        make_blocks.build(a=orm.Int(1))


def test_not_required_entry_keys_may_be_absent_from_source():
    """Sources missing ``NotRequired`` entry keys link, round-trip and run."""

    class Entry(TypedDict):
        value: orm.Int
        tag: orm.Int
        extra_a: NotRequired[orm.Int]
        extra_b: NotRequired[orm.Int]

    class OutA(TypedDict):
        value: orm.Int
        tag: orm.Int
        extra_a: orm.Int

    class OutB(TypedDict):
        value: orm.Int
        tag: orm.Int
        extra_b: orm.Int

    @task.graph
    def sub_a(x: orm.Int) -> OutA:
        return OutA(value=mk(x, 10).result, tag=mk(x, 100).result, extra_a=mk(x, 1).result)

    @task.graph
    def sub_b(x: orm.Int) -> OutB:
        return OutB(value=mk(x, 10).result, tag=mk(x, 100).result, extra_b=mk(x, 2).result)

    class Out(TypedDict):
        blocks: Annotated[dict, dynamic(Entry)]

    @task.graph
    def make_blocks(a: orm.Int, b: orm.Int) -> Out:
        return Out(blocks={'first': sub_a(a), 'second': sub_b(b)})

    wg = make_blocks.build(a=orm.Int(2), b=orm.Int(5))
    WorkGraph.from_dict(wg.to_dict())
    wg.run()

    blocks = wg.outputs.blocks
    assert blocks.first.value.value == 12
    assert blocks.first.tag.value == 102
    assert blocks.first.extra_a.value == 3
    # the entry key absent from the source simply carries no value
    assert blocks.first.extra_b.value is None
    assert blocks.second.value.value == 15
    assert blocks.second.extra_a.value is None
    assert blocks.second.extra_b.value == 7
