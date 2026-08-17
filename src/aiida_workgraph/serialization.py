from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from aiida_pythonjob.data.serializer import all_serializers
from aiida_pythonjob.utils import serialize_ports
from node_graph.serializer import SerializationAdapter
from node_graph.utils import resolve_tagged_values

# Socket identifiers whose declared type is a Python primitive. When the
# persisted value round-trips through AiiDA storage it becomes the matching
# ``orm.BaseType`` node (``orm.Float``, ``orm.Int``, ``orm.Str``,
# ``orm.Bool``); ``deserialize`` must strip that wrapper before the value
# reaches a user-written ``@task.graph`` body whose signature declared a
# primitive type.
_PRIMITIVE_SOCKET_IDENTIFIERS = frozenset(
    {
        'workgraph.float',
        'workgraph.int',
        'workgraph.string',
        'workgraph.bool',
    }
)


def _flatten_enums(value: Any) -> Any:
    """Replace ``Enum`` members with their bare ``.value``, recursively.

    ``general_serializer`` (aiida-pythonjob) has no serializer for ``Enum``
    members: a raw ``Enum`` -- or one nested in a dict/list/tuple -- reaches
    it and the whole submission fails at submit time with an opaque
    serialization error. Collapsing members to their ``.value`` here keeps
    the payload JSON-serializable so ``serialize_ports`` succeeds. The
    ``isinstance`` check also matches wrapt proxies (node-graph's
    ``TaggedValue``) wrapping an enum member.

    Flattening is one-way. Whether a body then receives the ``Enum`` it
    declared is the installed ``node_graph``'s call: the read side
    (``coerce_inputs_from_spec``) rebuilds a socket's value only from the
    ``structured_type`` descriptor its spec records, and not every
    ``node_graph`` records one for ``Enum``. Without it a function task's
    body receives the bare value and a ``@task.graph`` body the stored
    ``orm.Str``; with it both receive the member, a ``@task.graph`` body
    behind a ``TaggedValue``. A body that must work under either writes
    ``Color(getattr(c, 'value', c))``, never ``Color(c)`` and never
    ``c is Color.RED``. See
    ``tests/test_serializer.py::test_enum_arrival_follows_the_node_graph_capability``.

    ``set``/``frozenset`` are deliberately left untouched: a set fails in
    ``general_serializer`` regardless of its contents (not JSON-serializable,
    no registered serializer), so descending into one to flatten enums would
    not make it serializable -- see
    ``tests/test_serializer.py::test_flatten_leaves_sets_untouched``.
    """
    if isinstance(value, Enum):
        return _flatten_enums(value.value)
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        for k, v in value.items():
            flat_key = _flatten_enums(k)
            if flat_key in out:
                # Two distinct keys (e.g. an Enum member and its bare value,
                # or two members sharing a ``.value``) collapse to one; raise
                # rather than silently drop an entry.
                raise ValueError(f'Enum key flattening collision: multiple keys map to {flat_key!r}')
            out[flat_key] = _flatten_enums(v)
        return out
    if isinstance(value, list):
        return [_flatten_enums(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_flatten_enums(v) for v in value)
    return value


class AiidaSerializationAdapter(SerializationAdapter):
    id: str = 'aiida'
    name: str = 'AiiDA'

    def __init__(self, serializers: Optional[Dict[str, str]] = None, user: Any = None) -> None:
        self.serializers = serializers or all_serializers
        self.user = user

    def serialize(self, value: Any, socket: Any, *, store: bool) -> Any:
        if socket is None:
            return value
        spec = socket._to_spec()
        resolve_tagged_values(value)
        value = _flatten_enums(value)
        return serialize_ports(
            python_data=value,
            port_schema=spec,
            serializers=self.serializers,
            user=self.user,
        )

    def deserialize(self, value: Any, socket: Any) -> Any:
        """Unwrap ``orm.BaseType`` to its Python value for primitive sockets.

        Symmetric counterpart to ``serialize``: where the write path
        auto-serialises primitive inputs into ``orm.Float`` / ``orm.Int`` /
        ``orm.Str`` / ``orm.Bool`` for provenance, the read path (invoked
        just before a ``@task.graph`` body is called) hands the body the
        primitive that its signature actually declared. Non-primitive sockets
        (``workgraph.any``, AiiDA-typed sockets, etc.) pass through.

        Dataclass-typed sockets get the same unwrap applied to their
        fields. Without this the ``cls(**value)`` reconstruction in
        ``node_graph.utils.struct_utils.coerce_structured_value`` re-packs
        node-promoted scalars into the dataclass, so a field declared
        ``int`` lands as ``orm.Int`` and downstream stdlib calls
        (``range(self.ntyp)``, etc.) fail with ``TypeError`` because
        ``orm.Int`` doesn't implement ``__index__``.
        """
        from dataclasses import fields, is_dataclass, replace

        from aiida import orm

        if isinstance(value, orm.BaseType):
            identifier = getattr(socket, '_identifier', None)
            if identifier in _PRIMITIVE_SOCKET_IDENTIFIERS:
                return value.value
            return value

        # Same unwrap for container-typed sockets: a plain dict/list built in
        # a parent graph body is promoted to ``orm.Dict`` / ``orm.List`` for
        # provenance, but a body whose signature declares ``dict`` / ``list``
        # expects the plain container (e.g. ``get_protocol_inputs`` calls
        # ``overrides.copy()``, which ``orm.Dict`` doesn't implement).
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

    def serialize_ports(self, python_data: Any, port_schema: Any, *, store: bool) -> Any:
        resolve_tagged_values(python_data)
        python_data = _flatten_enums(python_data)
        return serialize_ports(
            python_data=python_data,
            port_schema=port_schema,
            serializers=self.serializers,
            user=self.user,
        )
