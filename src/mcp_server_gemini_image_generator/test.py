import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def test_caption():
    async with Client(StreamableHttpTransport("http://127.0.0.1:8002/mcp")) as client:

        # LinkedIn
        result = await client.call_tool(
            "caption_generator",
            {
                "platform": "linkedin",
                "brand": "Tesla",
                "prompt": "announcing new solar roof",
                "tone": "professional"
            }
        )
        print("LinkedIn Caption:", result)

        # Twitter
        result = await client.call_tool(
            "caption_generator",
            {
                "platform": "twitter",
                "brand": "Nike",
                "prompt": "new Air Max launch",
                "tone": "funny"
            }
        )
        print("Twitter Caption:", result)

if __name__ == "__main__":
    asyncio.run(test_caption())
