import asyncio

from browser_use import Agent, Browser, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

async def example():
    browser = Browser(
        # use_cloud=True,  # Uncomment to use a stealth browser on Browser Use Cloud
    )

    llm = ChatOpenAI( model="qwen/qwen3-vl-8b")

    agent = Agent(
        task="搜尋 台灣積體電路製造 的最新股價並總結相關新聞。",
        llm=llm,
        browser=browser,
    )

    history = await agent.run()
    return history

if __name__ == "__main__":
    history = asyncio.run(example())