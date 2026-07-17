from enum import Enum, IntEnum

from aiida_workgraph import WorkGraph, task
import pytest

from aiida_workgraph.serialization import AiidaSerializationAdapter, _flatten_enums


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


@task(outputs=['type_name', 'is_enum', 'eq_member'])
def _observe_enum(c: Color) -> dict:
    """Report what the body actually receives for an Enum-typed input."""
    return {
        'type_name': type(c).__name__,
        'is_enum': isinstance(c, Color),
        'eq_member': c == Color.RED,
    }


def test_body_receives_bare_value_not_member(aiida_profile):
    """Pin the real round-trip behaviour under the pinned ``node_graph``: a task
    body declaring a *plain* ``Enum`` input receives the BARE value, not the
    member. ``node_graph`` records ``structured_type`` extras only for
    dataclass/pydantic/TypedDict, so ``coerce_inputs_from_spec`` never rebuilds
    an ``Enum``. If node-graph later adds enum reconstruction, this test flips
    loudly and the docstring/PR claim must be revisited."""
    wg = WorkGraph('enum_roundtrip')
    t = wg.add_task(_observe_enum, name='observe', c=Color.RED)
    wg.run()
    assert wg.state == 'FINISHED'
    assert t.outputs['type_name'].value.value == 'str'
    assert t.outputs['is_enum'].value.value is False
    assert t.outputs['eq_member'].value.value is False
