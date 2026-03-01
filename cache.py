import redis
import json

r = redis.Redis(host="redis", port=6379, decode_responses=True)

def get_cache(query: str):
    return r.get(query)

def set_cache(query: str, value: str):
    r.set(query, value)