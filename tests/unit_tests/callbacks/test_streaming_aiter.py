
import unittest
import pytest
import asyncio

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

from langchain.llms.base import LLMResult,Generation


@pytest.mark.asyncio
async def test_streaming_aiter():
    siter = AsyncIteratorCallbackHandler()

    wait_event: asyncio.Event = asyncio.Event()
    done_event : asyncio.Event = asyncio.Event()

    async def run_producer():
        await siter.on_llm_new_token("a")
        await siter.on_llm_new_token("b")
        await siter.on_llm_new_token("c")
        await wait_event.wait()
        await siter.on_llm_new_token("d")
        await siter.on_llm_new_token("e")
        await siter.on_llm_new_token("f")

        ll = LLMResult( generations = [ [ Generation(text = "abcef") ] ])

        await siter.on_llm_end(response=ll)
        done_event.set()

    items = []

    async def run_consumer():
        async for x in siter.aiter():
            print(f"add item {x}")
            items.append(x)
            if x == "c":
                wait_event.set()
                await done_event.wait()

        return items
        
    await asyncio.wait(
        [
            asyncio.get_running_loop().create_task(run_consumer()),
            asyncio.get_running_loop().create_task(run_producer())
        ], 
        return_when=asyncio.ALL_COMPLETED
    )

    assert len(items) == 3

if __name__ == "__main__":
    unittest.main()