from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from aiida_pythonjob.data.serializer import all_serializers
from aiida_pythonjob.utils import serialize_ports
from node_graph.serializer import SerializationAdapter
from node_graph.utils import resolve_tagged_values


def _flatten_enums(value: Any) -> Any:
    """Replace ``Enum`` members with their bare ``.value``, recursively.

    ``node_graph`` declares that an enum-typed socket's serialized form is
    its bare value (``socket_spec`` records ``structured_type`` extras so
    ``coerce_inputs_from_spec`` can rebuild the member before a task body
    runs). The write path has to honour that contract: without this,
    ``serialize_ports`` hands the raw ``Enum`` instance to
    ``general_serializer``, which has no serializer for it and fails at
    submit time. ``isinstance`` also matches wrapt proxies (TaggedValue)
    around enum members.
    """
    if isinstance(value, Enum):
        return _flatten_enums(value.value)
    if isinstance(value, dict):
        return {_flatten_enums(k): _flatten_enums(v) for k, v in value.items()}
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

    def serialize_ports(self, python_data: Any, port_schema: Any, *, store: bool) -> Any:
        resolve_tagged_values(python_data)
        python_data = _flatten_enums(python_data)
        return serialize_ports(
            python_data=python_data,
            port_schema=port_schema,
            serializers=self.serializers,
            user=self.user,
        )
