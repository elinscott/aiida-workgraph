from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

from aiida_workgraph import WorkGraph, task
from aiida_workgraph.serialization import AiidaSerializationAdapter, _flatten_enums
import pytest


@task.graph()
def sub_workflow(func):
    func()


def test_func_as_input(capsys):
    from aiida_workgraph.executors.test import add

    wg = WorkGraph('test_func_as_input')
    wg.add_task(sub_workflow, func=add, name='sub_workflow')
    with pytest.raises(Exception, match='Cannot serialize the provided object'):
        wg.save()


class Color(Enum):
    """Plain enum whose value is a bare string."""

    RED = 'red'
    BLUE = 'blue'


class Priority(IntEnum):
    """Enum whose value is a bare int."""

    LOW = 1
    HIGH = 9


class Flavour(str, Enum):
    """Enum member that is itself a ``str`` subclass."""

    SWEET = 'sweet'
    SOUR = 'sour'


def test_flatten_bare_enum_member():
    """A lone member collapses to its ``.value``, not a mangled ``Color.RED``."""
    assert _flatten_enums(Color.RED) == 'red'
    assert not isinstance(_flatten_enums(Color.RED), Enum)


def test_flatten_intenum_and_str_enum_yield_bare_value():
    """The bare value's own type is preserved (int stays int, str stays str)."""
    assert _flatten_enums(Priority.HIGH) == 9
    assert type(_flatten_enums(Priority.HIGH)) is int
    assert _flatten_enums(Flavour.SWEET) == 'sweet'
    assert type(_flatten_enums(Flavour.SWEET)) is str


def test_flatten_dict_keys_and_values():
    """Enums appearing as dict keys *and* values both flatten."""
    assert _flatten_enums({Color.RED: Color.BLUE, 'n': Priority.LOW}) == {'red': 'blue', 'n': 1}


def test_flatten_list_and_tuple_preserve_container_type():
    """Sequences recurse element-wise and keep list-vs-tuple identity."""
    assert _flatten_enums([Color.RED, 1, 'x']) == ['red', 1, 'x']
    out = _flatten_enums((Color.RED, Priority.HIGH))
    assert out == ('red', 9)
    assert isinstance(out, tuple)


def test_flatten_mixed_nesting():
    """Enums buried in a dict/list/tuple mixture are all reached."""
    payload = {
        'a': [Color.RED, {'b': Priority.LOW}],
        Color.BLUE: ('x', Flavour.SOUR),
    }
    assert _flatten_enums(payload) == {
        'a': ['red', {'b': 1}],
        'blue': ('x', 'sour'),
    }


def test_flatten_passthrough_of_enum_free_payload():
    """An enum-free payload comes back with identical values."""
    payload = {'n': 3, 'items': [1, 'two', (3.0, None)], 'flag': True}
    assert _flatten_enums(payload) == payload


def test_flatten_wrapt_proxied_enum_member():
    """A wrapt ``ObjectProxy`` (as TaggedValue wraps members) still flattens."""
    wrapt = pytest.importorskip('wrapt')
    proxied = wrapt.ObjectProxy(Color.RED)
    assert _flatten_enums(proxied) == 'red'
    assert _flatten_enums({proxied: proxied}) == {'red': 'red'}


def test_flatten_dict_key_collision_raises():
    """Two distinct keys collapsing to the same flattened key raises rather than
    silently dropping an entry. Here ``Color.RED`` (a plain Enum, so distinct in
    identity and hash from the bare string) and ``'red'`` are separate keys that
    both flatten to ``'red'``."""
    payload = {Color.RED: 1, 'red': 2}
    assert len(payload) == 2  # distinct keys before flattening
    with pytest.raises(ValueError, match='collision'):
        _flatten_enums(payload)


def test_flatten_leaves_sets_untouched():
    """``set``/``frozenset`` are not descended into: they fail in
    ``general_serializer`` regardless of contents, so flattening enums inside
    them would not make them serializable. Pin that they pass through unchanged
    (still holding Enum members)."""
    s = {Color.RED, Color.BLUE}
    out = _flatten_enums(s)
    assert out is s
    assert Color.RED in out
    fs = frozenset({Priority.LOW})
    assert _flatten_enums(fs) is fs


def test_serialize_ports_accepts_enum_value(aiida_profile):
    """End-to-end through the adapter: an enum-valued namespace entry serializes
    without raising, landing as the flattened bare value (needs a profile so the
    downstream ``general_serializer`` can build the node)."""
    from aiida import orm
    from node_graph.socket_spec import SocketMeta, SocketSpec

    spec = SocketSpec(identifier='node_graph.namespace', meta=SocketMeta(dynamic=True))
    out = AiidaSerializationAdapter().serialize_ports({'c': Color.RED, 'n': 3}, spec, store=False)
    assert isinstance(out['c'], orm.Str)
    assert out['c'].value == 'red'


@task(outputs=['type_name', 'is_member', 'rebuilt_name'])
def observe_enum(c: Color) -> dict:
    """Report what a function task's body receives for an Enum-typed input."""
    return {
        'type_name': type(c).__name__,
        'is_member': isinstance(c, Color),
        'rebuilt_name': Color(getattr(c, 'value', c)).name,
    }


@task()
def echo_str(s: str) -> str:
    return s


@task.graph()
def observe_enum_in_graph(c: Color) -> str:
    """Rebuild the member from what a ``@task.graph`` body receives."""
    return echo_str(s=Color(getattr(c, 'value', c)).name)


def _node_graph_rebuilds_enums() -> bool:
    """Return whether the installed ``node_graph`` reconstructs Enum sockets.

    ``coerce_inputs_from_spec`` rebuilds a socket's value only from the
    ``structured_type`` descriptor its spec records, so a descriptor for an
    ``Enum`` type is exactly the capability.
    """
    from node_graph.utils.struct_utils import structured_type_info

    return structured_type_info(Color) is not None


def test_enum_input_rebuilds_to_the_member_that_was_passed(aiida_profile):
    """``_flatten_enums`` loses nothing the member cannot be rebuilt from:
    ``Color(getattr(c, 'value', c))`` returns ``Color.RED`` in both a function
    task's body and a ``@task.graph`` body, whatever form the boundary
    delivered (the flattened ``'red'``, the stored ``orm.Str`` behind a
    ``TaggedValue``, or the member behind one)."""
    wg = WorkGraph('enum_roundtrip')
    fn = wg.add_task(observe_enum, name='observe', c=Color.RED)
    gr = wg.add_task(observe_enum_in_graph, name='observe_graph', c=Color.RED)
    wg.run()
    assert wg.state == 'FINISHED'
    assert fn.outputs['rebuilt_name'].value.value == 'RED'
    assert gr.outputs['result'].value.value == 'RED'


def test_enum_arrival_follows_the_node_graph_capability(aiida_profile):
    """Which form arrives is the installed ``node_graph``'s call, not this
    package's: the ``Enum`` member when its specs carry a ``structured_type``
    descriptor for ``Enum``, the flattened value otherwise. Asserting the two
    agree pins the packages to each other rather than to a version, so a
    reconstruction that stops working still fails here."""
    wg = WorkGraph('enum_arrival')
    t = wg.add_task(observe_enum, name='observe', c=Color.RED)
    wg.run()
    assert wg.state == 'FINISHED'
    rebuilds = _node_graph_rebuilds_enums()
    assert t.outputs['is_member'].value.value is rebuilds
    assert t.outputs['type_name'].value.value == ('Color' if rebuilds else 'str')


# --- deserialize must not break provenance while unwrapping ------------------
#
# A ``@task.graph`` body whose signature declares a primitive type receives
# the unwrapped ``str``/``int``/``dict``/``list``/dataclass, not the
# ``orm.BaseType``/``orm.Dict``/``orm.List`` node the write path promoted it
# to. If a sub-task inside the body is then bound from that value, the value
# must still carry the ``TaggedValue`` tag that draws a link back to the
# graph input's own node -- otherwise the sub-task stores a brand-new,
# unlinked copy (an orphan input with zero incoming links) instead of
# referencing the graph input.


@task()
def echo_task(label, count):
    return f'{label}-{count}'


@task.graph()
def bare_graph(label, count):
    return echo_task(label=label, count=count)


@task.graph()
def annotated_graph(label: str, count: int):
    return echo_task(label=label, count=count)


@task.graph()
def annotated_any_graph(label: Any, count: Any):
    return echo_task(label=label, count=count)


def _label_link_uuids(node):
    """Return (graph-input node, sub-task input node) for the ``label`` link."""
    outer = inner = None
    for link in node.process.base.links.get_incoming().all():
        if link.link_label.endswith('label'):
            outer = link.node
    for child in node.process.called:
        for link in child.base.links.get_incoming().all():
            if link.link_label.endswith('label'):
                inner = link.node
    return outer, inner


@pytest.mark.parametrize(
    'entry, name', [(bare_graph, 'bare'), (annotated_graph, 'annotated'), (annotated_any_graph, 'annotated_any')]
)
def test_primitive_graph_input_link_survives_deserialize(entry, name):
    """A primitive graph input still links to the sub-task it feeds.

    ``annotated`` is the case #799 broke: ``label: str`` makes ``deserialize``
    unwrap the socket's ``orm.Str`` to a bare ``str`` before the body runs, and
    without tag preservation that bare ``str`` becomes a fresh, unlinked node
    when ``echo_task`` is bound from it.
    """
    wg = WorkGraph(f'link_{name}')
    node = wg.add_task(entry, name='g', label='silicon', count=2)
    wg.run()
    outer, inner = _label_link_uuids(node)
    assert outer is not None and inner is not None
    assert outer.uuid == inner.uuid, (
        f'sub-task input {inner.uuid} is an orphan copy of graph input {outer.uuid}, '
        f'not a link to it (incoming links: {len(inner.base.links.get_incoming().all())})'
    )


@task()
def echo_container_task(d, lst):
    return {'d': d, 'l': lst}


@task.graph()
def container_graph(d: dict, lst: list):
    return echo_container_task(d=d, lst=lst)


def test_dict_and_list_graph_input_link_survives_deserialize():
    """A dict/list graph input, unwrapped from ``orm.Dict``/``orm.List``, still links."""
    wg = WorkGraph('link_container')
    node = wg.add_task(container_graph, name='g', d={'a': 1}, lst=[1, 2])
    wg.run()
    outer = inner = None
    for link in node.process.base.links.get_incoming().all():
        if link.link_label.endswith('d'):
            outer = link.node
    for child in node.process.called:
        for link in child.base.links.get_incoming().all():
            if link.link_label.endswith('d'):
                inner = link.node
    assert outer is not None and inner is not None
    assert outer.uuid == inner.uuid, 'dict graph input was copied instead of linked'


@dataclass
class _Named:
    label: str
    count: int


@task.graph()
def dataclass_graph(data: _Named):
    return echo_task(label=data.label, count=data.count)


def test_dataclass_field_link_survives_deserialize():
    """A dataclass-typed graph input's fields are unwrapped without losing their links.

    Each field is tagged independently of the dataclass instance (a
    structured socket is tagged leaf by leaf), so the fix must preserve the
    tag on the field value it unwraps, not on the dataclass instance itself.
    """
    wg = WorkGraph('link_dataclass')
    node = wg.add_task(dataclass_graph, name='g', data=_Named(label='silicon', count=2))
    wg.run()
    outer, inner = _label_link_uuids(node)
    assert outer is not None and inner is not None
    assert outer.uuid == inner.uuid, 'dataclass field was copied instead of linked'


def _pre_fix_deserialize(self, value, socket):
    """The #799 body before this fix: unwraps via attribute access with no
    awareness of ``TaggedValue``, so the tag that draws a provenance link is
    silently dropped. Used as a negative control below."""
    from dataclasses import fields, is_dataclass, replace

    from aiida import orm

    if isinstance(value, orm.BaseType):
        identifier = getattr(socket, '_identifier', None)
        if identifier in {'workgraph.float', 'workgraph.int', 'workgraph.string', 'workgraph.bool'}:
            return value.value
        return value

    identifier = getattr(socket, '_identifier', None)
    if isinstance(value, orm.Dict) and identifier == 'workgraph.dict':
        return value.get_dict()
    if isinstance(value, orm.List) and identifier == 'workgraph.list':
        return value.get_list()

    if is_dataclass(value) and not isinstance(value, type):
        field_updates = {
            f.name: getattr(value, f.name).value
            for f in fields(value)
            if isinstance(getattr(value, f.name), orm.BaseType)
        }
        if field_updates:
            return replace(value, **field_updates)

    return value


def test_negative_control_pre_fix_deserialize_loses_the_link(monkeypatch):
    """Discriminating control: the pre-fix unwrap reproduces #799's link loss.

    If this failed to reproduce the loss, the provenance assertions above
    would prove nothing -- they could be passing for a reason unrelated to
    the fix.
    """
    monkeypatch.setattr(AiidaSerializationAdapter, 'deserialize', _pre_fix_deserialize)
    wg = WorkGraph('link_negative_control')
    node = wg.add_task(annotated_graph, name='g', label='silicon', count=2)
    wg.run()
    outer, inner = _label_link_uuids(node)
    assert outer is not None and inner is not None
    assert outer.uuid != inner.uuid, 'pre-fix deserialize unexpectedly preserved the link'
    assert len(inner.base.links.get_incoming().all()) == 0, 'expected an orphan copy'
