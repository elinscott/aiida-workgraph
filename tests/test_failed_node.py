import pytest
from typing import Callable


@pytest.mark.usefixtures('started_daemon_client')
def test_failed_node(decorated_sqrt: Callable, decorated_add: Callable) -> None:
    """Submit simple calcfunction."""
    from aiida_workgraph import WorkGraph
    from aiida.orm import Float

    wg = WorkGraph(name='test_failed_node')
    wg.add_task(decorated_add, 'add1', x=Float(1), y=Float(2))
    sqrt1 = wg.add_task(decorated_sqrt, 'sqrt1', x=Float(-1))
    wg.add_task(decorated_sqrt, 'sqrt2', x=sqrt1.outputs.result)
    wg.submit(wait=True)
    # print("results: ", results[])
    assert wg.process.exit_status == 302
    assert (
        wg.process.exit_message
        == "WorkGraph finished, but tasks: ['sqrt1'] failed. Thus all their child tasks are skipped."
    )


@pytest.mark.parametrize('decorator_name', ['calcfunction', 'workfunction'])
def test_get_task_process_for_excepted_function_task(decorator_name: str) -> None:
    """A raising calcfunction/workfunction still leaves its process reference retrievable.

    Regression test for aiidateam/aiida-workgraph#809: ``run_get_node`` stores the
    excepted node and CALL-links it before re-raising the wrapped function's
    exception, but the engine used to lose that reference on the way out, so
    ``get_task_process`` returned ``None`` for a task that provenance shows did run.
    """
    from aiida.orm import Int, ProcessNode

    from aiida_workgraph import WorkGraph, task
    from aiida_workgraph.orm.utils import deserialize_safe

    decorator = getattr(task, decorator_name)

    @decorator()
    def fails(x):
        raise ValueError('deliberate failure')
        return x  # unreachable; gives the task's outputs a 'result' socket to link from

    @decorator()
    def downstream(x):
        return x

    wg = WorkGraph(name=f'test_excepted_{decorator_name}_process_reference')
    fail_task = wg.add_task(fails, 'fail1', x=Int(1))
    wg.add_task(downstream, 'after1', x=fail_task.outputs.result)
    wg.run()

    node = wg.process
    assert node.exit_status == 302
    assert node.is_finished_ok is False
    assert node.get_task_state('fail1') == 'FAILED'
    assert node.get_task_state('after1') == 'SKIPPED'
    skipped_serialized = node.get_task_process('after1')
    assert skipped_serialized is None or deserialize_safe(skipped_serialized) is None

    serialized = node.get_task_process('fail1')
    assert serialized is not None
    excepted_process = deserialize_safe(serialized)
    assert isinstance(excepted_process, ProcessNode)
    assert excepted_process.process_state.value.upper() == 'EXCEPTED'
    # the recovered node is the same one CALL-linked into provenance
    assert [child.pk for child in node.called] == [excepted_process.pk]
