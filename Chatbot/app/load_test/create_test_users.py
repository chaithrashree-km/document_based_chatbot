"""
Pre-create 150 test users for load testing.
Run this ONCE before running locust.

Usage:
    python create_test_users.py --host http://localhost:8000 --count 150
"""

import httpx
import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

async def create_user(client: httpx.AsyncClient, host: str, i: int):
    payload = {
        "username": f"loadtest_user_{i}",
        "email": f"loadtest_user_{i}@test.com",
        "password": "LoadTest@123"
    }
    try:
        r = await client.post(f"{host}/signup", json=payload, timeout=10)
        if r.status_code == 200:
            logging.info(f"Created user {i}")
        elif r.status_code == 400 and "already registered" in r.text:
            logging.info(f"User {i} already exists — skipping")
        else:
            logging.warning(f"User {i} failed: {r.status_code} {r.text}")
    except Exception as e:
        logging.error(f"User {i} error: {e}")

async def main(host: str, count: int):
    # SlowAPI limits signup to 10/minute — batch with delays
    async with httpx.AsyncClient() as client:
        batch_size = 8   # stay under the 10/minute rate limit
        for batch_start in range(1, count + 1, batch_size):
            batch = range(batch_start, min(batch_start + batch_size, count + 1))
            await asyncio.gather(*[create_user(client, host, i) for i in batch])
            if batch_start + batch_size <= count:
                logging.info("Waiting 65s for rate limit window to reset...")
                await asyncio.sleep(65)

    logging.info(f"Done! {count} test users created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.count))
