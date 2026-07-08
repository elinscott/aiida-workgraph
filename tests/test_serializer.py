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
