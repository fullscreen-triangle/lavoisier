"""
Caching utilities for storing and retrieving intermediate results
"""
import os
import time
import hashlib
import pickle
import gzip
import logging
import shutil
import sys
from typing import Any, Dict, Optional, Tuple, Callable
from functools import wraps
from pathlib import Path
import threading
import weakref

from lavoisier.core.config import GlobalConfig

# Logger for caching operations
logger = logging.getLogger(__name__)


class CacheItem:
    """Represents a single cached item with metadata"""
    def __init__(self, key: str, value: Any, level: str, ttl_minutes: int = 60):
        self.key = key
        self.value = value
        self.level = level
        self.timestamp = time.time()
        self.ttl_minutes = ttl_minutes
        self.size_bytes = self._estimate_size()
    
    def _estimate_size(self) -> int:
        """Estimate the size of the value in bytes"""
        try:
            # Use pickle to estimate size
            data = pickle.dumps(self.value)
            return len(data)
        except:
            # Fallback if object can't be pickled
            return sys.getsizeof(self.value)
    
    def is_expired(self) -> bool:
        """Check if the item has expired based on TTL"""
        elapsed_minutes = (time.time() - self.timestamp) / 60
        return elapsed_minutes > self.ttl_minutes


class Cache:
    """Base class for caching implementations"""
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.enabled = config.caching.enabled
        self.ttl_minutes = config.caching.ttl_minutes
        self.compress = config.caching.compress
        self.levels = config.caching.levels
    
    def get(self, key: str, level: str = "processed") -> Tuple[bool, Any]:
        """Get item from cache, returns (success, value)"""
        if not self.enabled or level not in self.levels:
            return False, None
        return self._get_impl(key, level)
    
    def set(self, key: str, value: Any, level: str = "processed") -> bool:
        """Set item in cache, returns success"""
        if not self.enabled or level not in self.levels:
            return False
        return self._set_impl(key, value, level)
    
    def invalidate(self, key: str, level: Optional[str] = None) -> bool:
        """Remove item from cache, returns success"""
        if not self.enabled:
            return False
        return self._invalidate_impl(key, level)
    
    def clear(self, level: Optional[str] = None) -> bool:
        """Clear all items from cache or from specific level, returns success"""
        if not self.enabled:
            return False
        return self._clear_impl(level)
    
    def _get_impl(self, key: str, level: str) -> Tuple[bool, Any]:
        """Implementation of get operation, to be overridden"""
        raise NotImplementedError()
    
    def _set_impl(self, key: str, value: Any, level: str) -> bool:
        """Implementation of set operation, to be overridden"""
        raise NotImplementedError()
    
    def _invalidate_impl(self, key: str, level: Optional[str]) -> bool:
        """Implementation of invalidate operation, to be overridden"""
        raise NotImplementedError()
    
    def _clear_impl(self, level: Optional[str]) -> bool:
        """Implementation of clear operation, to be overridden"""
        raise NotImplementedError()
    
    @staticmethod
    def generate_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate a cache key from a function name and its arguments"""
        # Create a string representation of the arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        
        # Create a hash of the function name and arguments
        key = hashlib.md5(f"{func_name}:{args_str}".encode()).hexdigest()
        return key


class MemoryCache(Cache):
    """In-memory cache implementation"""
    def __init__(self, config: GlobalConfig):
        super().__init__(config)
        self.max_size_bytes = config.caching.max_memory_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.cache: Dict[str, Dict[str, CacheItem]] = {level: {} for level in self.levels}
        self.lock = threading.RLock()
    
    def _get_impl(self, key: str, level: str) -> Tuple[bool, Any]:
        """Get item from memory cache"""
        with self.lock:
            if level not in self.cache or key not in self.cache[level]:
                return False, None
            
            item = self.cache[level][key]
            
            # Check if item is expired
            if item.is_expired():
                self._remove_item(key, level)
                return False, None
            
            return True, item.value
    
    def _set_impl(self, key: str, value: Any, level: str) -> bool:
        """Set item in memory cache"""
        with self.lock:
            # Create new cache item
            item = CacheItem(key, value, level, self.ttl_minutes)
            
            # Check if adding this item would exceed max size
            if item.size_bytes > self.max_size_bytes:
                logger.warning(f"Cache item too large to store: {item.size_bytes} bytes")
                return False
            
            # Make room for new item if needed
            while self.current_size_bytes + item.size_bytes > self.max_size_bytes:
                if not self._evict_oldest_item():
                    # Nothing left to evict
                    return False
            
            # Add the item to cache
            if key in self.cache[level]:
                old_item = self.cache[level][key]
                self.current_size_bytes -= old_item.size_bytes
            
            self.cache[level][key] = item
            self.current_size_bytes += item.size_bytes
            
            return True
    
    def _invalidate_impl(self, key: str, level: Optional[str]) -> bool:
        """Remove item from memory cache"""
        with self.lock:
            if level is None:
                # Remove from all levels
                found = False
                for lvl in self.levels:
                    if key in self.cache[lvl]:
                        self._remove_item(key, lvl)
                        found = True
                return found
            else:
                # Remove from specific level
                if level in self.cache and key in self.cache[level]:
                    self._remove_item(key, level)
                    return True
            return False
    
    def _clear_impl(self, level: Optional[str]) -> bool:
        """Clear all items from memory cache"""
        with self.lock:
            if level is None:
                # Clear all levels
                self.cache = {level: {} for level in self.levels}
                self.current_size_bytes = 0
            else:
                # Clear specific level
                if level in self.cache:
                    # Subtract size of items in this level
                    for key, item in self.cache[level].items():
                        self.current_size_bytes -= item.size_bytes
                    self.cache[level] = {}
            return True
    
    def _evict_oldest_item(self) -> bool:
        """Evict the oldest item from the cache"""
        with self.lock:
            oldest_time = float('inf')
            oldest_key = None
            oldest_level = None
            
            # Find the oldest item across all levels
            for level in self.levels:
                for key, item in self.cache[level].items():
                    if item.timestamp < oldest_time:
                        oldest_time = item.timestamp
                        oldest_key = key
                        oldest_level = level
            
            # Evict the oldest item if found
            if oldest_key is not None:
                self._remove_item(oldest_key, oldest_level)
                return True
            
            return False
    
    def _remove_item(self, key: str, level: str) -> None:
        """Remove an item from the cache and update size"""
        if key in self.cache[level]:
            item = self.cache[level][key]
            self.current_size_bytes -= item.size_bytes
            del self.cache[level][key]


class DiskCache(Cache):
    """Disk-based cache implementation"""
    def __init__(self, config: GlobalConfig):
        super().__init__(config)
        self.cache_dir = Path(config.caching.disk_cache_path)
        self.lock = threading.RLock()
        
        # Ensure cache directory exists
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory structure exists"""
        with self.lock:
            # Create main cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Create level subdirectories
            for level in self.levels:
                level_dir = self.cache_dir / level
                os.makedirs(level_dir, exist_ok=True)
    
    def _get_item_path(self, key: str, level: str) -> Path:
        """Get path for a cached item"""
        return self.cache_dir / level / f"{key}.{'gz' if self.compress else 'pkl'}"
    
    def _get_metadata_path(self, key: str, level: str) -> Path:
        """Get path for a cached item's metadata"""
        return self.cache_dir / level / f"{key}.meta"
    
    def _get_impl(self, key: str, level: str) -> Tuple[bool, Any]:
        """Get item from disk cache"""
        with self.lock:
            item_path = self._get_item_path(key, level)
            meta_path = self._get_metadata_path(key, level)
            
            # Check if files exist
            if not item_path.exists() or not meta_path.exists():
                return False, None
            
            try:
                # Read metadata
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Check if item is expired
                if (time.time() - metadata['timestamp']) / 60 > self.ttl_minutes:
                    # Remove expired item
                    item_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return False, None
                
                # Read item
                if self.compress:
                    with gzip.open(item_path, 'rb') as f:
                        value = pickle.load(f)
                else:
                    with open(item_path, 'rb') as f:
                        value = pickle.load(f)
                
                return True, value
                
            except Exception as e:
                logger.error(f"Error reading cache item {key} from disk: {str(e)}")
                
                # Clean up corrupt files
                try:
                    item_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                except:
                    pass
                
                return False, None
    
    def _set_impl(self, key: str, value: Any, level: str) -> bool:
        """Set item in disk cache"""
        with self.lock:
            item_path = self._get_item_path(key, level)
            meta_path = self._get_metadata_path(key, level)
            
            try:
                # Ensure level directory exists
                level_dir = self.cache_dir / level
                os.makedirs(level_dir, exist_ok=True)
                
                # Write item
                if self.compress:
                    with gzip.open(item_path, 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(item_path, 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Write metadata
                metadata = {
                    'key': key,
                    'level': level,
                    'timestamp': time.time(),
                    'ttl_minutes': self.ttl_minutes
                }
                
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                return True
                
            except Exception as e:
                logger.error(f"Error writing cache item {key} to disk: {str(e)}")
                
                # Clean up partial files
                try:
                    item_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                except:
                    pass
                
                return False
    
    def _invalidate_impl(self, key: str, level: Optional[str]) -> bool:
        """Remove item from disk cache"""
        with self.lock:
            found = False
            
            if level is None:
                # Remove from all levels
                for lvl in self.levels:
                    item_path = self._get_item_path(key, lvl)
                    meta_path = self._get_metadata_path(key, lvl)
                    
                    if item_path.exists() or meta_path.exists():
                        item_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        found = True
            else:
                # Remove from specific level
                item_path = self._get_item_path(key, level)
                meta_path = self._get_metadata_path(key, level)
                
                if item_path.exists() or meta_path.exists():
                    item_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    found = True
            
            return found
    
    def _clear_impl(self, level: Optional[str]) -> bool:
        """Clear all items from disk cache"""
        with self.lock:
            try:
                if level is None:
                    # Clear all levels
                    for lvl in self.levels:
                        level_dir = self.cache_dir / lvl
                        if level_dir.exists():
                            shutil.rmtree(level_dir)
                            os.makedirs(level_dir, exist_ok=True)
                else:
                    # Clear specific level
                    level_dir = self.cache_dir / level
                    if level_dir.exists():
                        shutil.rmtree(level_dir)
                        os.makedirs(level_dir, exist_ok=True)
                
                return True
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                return False


class HybridCache(Cache):
    """Hybrid memory+disk cache implementation"""
    def __init__(self, config: GlobalConfig):
        super().__init__(config)
        self.memory_cache = MemoryCache(config)
        self.disk_cache = DiskCache(config)
    
    def _get_impl(self, key: str, level: str) -> Tuple[bool, Any]:
        """Get item from hybrid cache, checking memory first, then disk"""
        # Check memory cache first
        success, value = self.memory_cache._get_impl(key, level)
        if success:
            return True, value
        
        # Check disk cache if not in memory
        success, value = self.disk_cache._get_impl(key, level)
        if success:
            # Cache in memory for faster access next time
            self.memory_cache._set_impl(key, value, level)
            return True, value
        
        return False, None
    
    def _set_impl(self, key: str, value: Any, level: str) -> bool:
        """Set item in hybrid cache, storing in both memory and disk"""
        # Store in memory cache
        memory_success = self.memory_cache._set_impl(key, value, level)
        
        # Always store in disk cache
        disk_success = self.disk_cache._set_impl(key, value, level)
        
        return memory_success or disk_success
    
    def _invalidate_impl(self, key: str, level: Optional[str]) -> bool:
        """Remove item from hybrid cache"""
        memory_success = self.memory_cache._invalidate_impl(key, level)
        disk_success = self.disk_cache._invalidate_impl(key, level)
        
        return memory_success or disk_success
    
    def _clear_impl(self, level: Optional[str]) -> bool:
        """Clear all items from hybrid cache"""
        memory_success = self.memory_cache._clear_impl(level)
        disk_success = self.disk_cache._clear_impl(level)
        
        return memory_success and disk_success


def get_cache(config: GlobalConfig) -> Cache:
    """Factory function to create the appropriate cache based on config"""
    strategy = config.caching.strategy.lower()
    
    if not config.caching.enabled or strategy == "none":
        # Create a base cache that will always return misses
        return Cache(config)
    
    if strategy == "memory":
        return MemoryCache(config)
    elif strategy == "disk":
        return DiskCache(config)
    elif strategy == "memory_disk":
        return HybridCache(config)
    else:
        logger.warning(f"Unknown caching strategy: {strategy}, defaulting to hybrid")
        return HybridCache(config)


def cached(level: str = "processed"):
    """
    Decorator for caching function results
    
    Args:
        level: Cache level to use
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Skip caching if not enabled in the instance's config
            if not hasattr(self, 'config') or not hasattr(self.config, 'caching') or not self.config.caching.enabled:
                return func(self, *args, **kwargs)
            
            # Get or create cache
            if not hasattr(self, '_cache'):
                self._cache = get_cache(self.config)
            
            # Generate cache key
            key = Cache.generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            success, value = self._cache.get(key, level)
            if success:
                logger.debug(f"Cache hit for {func.__name__} at level {level}")
                return value
            
            # Cache miss, call function and store result
            logger.debug(f"Cache miss for {func.__name__} at level {level}")
            result = func(self, *args, **kwargs)
            
            # Store in cache
            self._cache.set(key, result, level)
            
            return result
        
        return wrapper
    
    return decorator


# Add a module-level function to ensure imports work correctly
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage stats in MB"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / (1024 * 1024),
        "vms": memory_info.vms / (1024 * 1024)
    } 