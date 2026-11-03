"""
Redis Caching Layer for Recommendation Engine.
Implements:
- Cache-aside pattern for personalized and cold-start recommendations.
- Configurable TTL expiration (default: 300 seconds).
- Automatic failover to an in-memory LRU cache when Redis daemon is unavailable.
- Real-time cache hit / miss telemetry.
"""

import os
import time
import json
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class InMemoryCache:
    def __init__(self, default_ttl=300):
        self.store = {}
        self.expiries = {}
        self.default_ttl = default_ttl

    def get(self, key):
        if key in self.store:
            if time.time() < self.expiries.get(key, float("inf")):
                return self.store[key]
            else:
                del self.store[key]
                if key in self.expiries:
                    del self.expiries[key]
        return None

    def set(self, key, value, ex=None):
        ttl = ex if ex is not None else self.default_ttl
        self.store[key] = value
        self.expiries[key] = time.time() + ttl
        return True

    def delete(self, key):
        self.store.pop(key, None)
        self.expiries.pop(key, None)
        return True

    def flush(self):
        self.store.clear()
        self.expiries.clear()
        return True

class RecommendationCache:
    def __init__(self, host=None, port=None, db=0, default_ttl=300):
        self.default_ttl = default_ttl
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        self.port = port or int(os.environ.get("REDIS_PORT", 6379))
        self.db = db

        self.client = None
        self.hits = 0
        self.misses = 0
        self.using_in_memory = False

        if REDIS_AVAILABLE:
            try:
                r = redis.Redis(host=self.host, port=self.port, db=self.db, socket_connect_timeout=0.5)
                r.ping()
                self.client = r
                logger.info(f"Connected to Redis server at {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis ({e}). Using in-memory fallback cache.")
                self.using_in_memory = True
                self.client = InMemoryCache(default_ttl=default_ttl)
        else:
            self.using_in_memory = True
            self.client = InMemoryCache(default_ttl=default_ttl)

    def _build_user_key(self, user_id, top_k, model_name="dcn"):
        return f"rec:user:{user_id}:topk:{top_k}:{model_name}"

    def _build_cold_key(self, genres, top_k):
        genres_str = "_".join(sorted(genres)) if genres else "popular"
        return f"rec:cold:{genres_str}:topk:{top_k}"

    def get_user_recommendations(self, user_id, top_k=10, model_name="dcn"):
        key = self._build_user_key(user_id, top_k, model_name)
        return self.get(key)

    def set_user_recommendations(self, user_id, recommendations, top_k=10, model_name="dcn", ttl=None):
        key = self._build_user_key(user_id, top_k, model_name)
        self.set(key, recommendations, ttl=ttl)

    def get_cold_recommendations(self, genres=None, top_k=10):
        key = self._build_cold_key(genres, top_k)
        return self.get(key)

    def set_cold_recommendations(self, recommendations, genres=None, top_k=10, ttl=None):
        key = self._build_cold_key(genres, top_k)
        self.set(key, recommendations, ttl=ttl)

    def get(self, key):
        try:
            val = self.client.get(key)
            if val is not None:
                self.hits += 1
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                return json.loads(val)
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.misses += 1
            return None

    def set(self, key, value, ttl=None):
        try:
            ex = ttl or self.default_ttl
            payload = json.dumps(value)
            if self.using_in_memory:
                self.client.set(key, payload, ex=ex)
            else:
                self.client.set(key, payload, ex=ex)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def invalidate_user(self, user_id):
        try:
            if self.using_in_memory:
                keys_to_del = [k for k in self.client.store.keys() if f"rec:user:{user_id}:" in k]
                for k in keys_to_del:
                    self.client.delete(k)
            else:
                for key in self.client.scan_iter(f"rec:user:{user_id}:*"):
                    self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return False

    def clear_all(self):
        try:
            if self.using_in_memory:
                self.client.flush()
            else:
                self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self):
        total = self.hits + self.misses
        hit_ratio = (self.hits / total) if total > 0 else 0.0
        return {
            "backend": "in_memory_lru" if self.using_in_memory else "redis",
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_ratio": round(hit_ratio, 4)
        }

