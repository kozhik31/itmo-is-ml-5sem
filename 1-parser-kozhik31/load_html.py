import os, asyncio, random, re, time
import aiohttp
import aiofiles
from aiohttp import ClientTimeout

TEMPLATES_DIR = "templates"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
PARAMS = {"l": "english", "cc": "us"}
COOKIES = {"birthtime": "568022401", "lastagecheckage": "1-January-1988", "mature_content": "1"}

CONCURRENCY = 30
RETRIES = 3
TIMEOUT = ClientTimeout(total=30)

async def fetch_one(session: aiohttp.ClientSession, app_id: str, sem: asyncio.Semaphore):
    path = os.path.join(TEMPLATES_DIR, f"{app_id}.html")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return app_id, None

    url = f"https://store.steampowered.com/app/{app_id}/"
    last_err = None
    for attempt in range(1, RETRIES + 1):
        async with sem:
            try:
                async with session.get(url, headers=HEADERS, params=PARAMS, cookies=COOKIES, timeout=TIMEOUT) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        if content:
                            async with aiofiles.open(path, "wb") as f:
                                await f.write(content)
                            await asyncio.sleep(random.uniform(0.02, 0.08))
                            return app_id, None
                    last_err = f"HTTP {resp.status}"
            except Exception as e:
                last_err = str(e)

        await asyncio.sleep((2 ** (attempt - 1)) + random.uniform(0.05, 0.3))

    return app_id, last_err

def read_ids(path="ids.txt"):
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    ids = [x for x in ids if re.fullmatch(r"\d+", x)]
    random.shuffle(ids)
    return ids

async def main_async():
    ids = read_ids()
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    ok = fail = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_one(session, app_id, sem) for app_id in ids]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            app_id, err = await coro
            if err is None:
                ok += 1
            else:
                fail += 1
                if fail <= 20 or fail % 100 == 0:
                    print(f"[warn] {app_id}: {err}")

            if i % 200 == 0:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                print(f"[progress] {i}/{len(ids)} done | ok={ok}, fail={fail} | {rate:.1f} req/s")

    print(f"Done: ok={ok}, fail={fail}")

if __name__ == "__main__":
    asyncio.run(main_async())
