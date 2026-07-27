import asyncio

from app.async_tasks import cancel_and_wait


def test_cancel_and_wait_stops_unfinished_request_work():
    async def exercise():
        canceled = asyncio.Event()

        async def request_work():
            try:
                await asyncio.Future()
            finally:
                canceled.set()

        task = asyncio.create_task(request_work())
        await asyncio.sleep(0)
        await cancel_and_wait(task)
        return task, canceled

    task, canceled = asyncio.run(exercise())

    assert task.cancelled()
    assert canceled.is_set()


def test_cancel_and_wait_observes_completed_request_work():
    async def exercise():
        task = asyncio.create_task(asyncio.sleep(0, result="complete"))
        await task
        await cancel_and_wait(task)
        return task

    task = asyncio.run(exercise())

    assert task.result() == "complete"
