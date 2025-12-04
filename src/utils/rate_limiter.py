# -*- coding: utf-8 -*-
"""
Thread-Safe Rate Limiter for Together.ai API (FIXED VERSION)

Handles 3000 RPM limit with thread-safe token bucket pattern.
Prevents API throttling during parallel relation extraction.

FIXES (Dec 4, 2025):
- Changed Lock to RLock for recursive acquire() calls
- Added better error handling

Author: Pau Barba i Colomer
Date: Dec 4, 2025
"""

import time
import threading
from collections import deque
from typing import Optional


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.
    
    Designed for Together.ai's 3000 requests/minute limit.
    Uses conservative buffer (2900 RPM) to avoid edge case throttling.
    
    THREAD SAFETY:
    - Uses RLock (reentrant lock) for recursive acquire() calls
    - All operations are atomic within lock context
    
    Usage:
        limiter = RateLimiter(max_calls_per_minute=2900)
        
        # Before each API call
        limiter.acquire()  # Blocks if rate limit reached
        response = api.call(...)
    """
    
    def __init__(self, max_calls_per_minute: int = 2900):
        """
        Initialize rate limiter.
        
        Args:
            max_calls_per_minute: Maximum calls allowed per 60-second window.
                Default 2900 (100 buffer below Together.ai's 3000 limit)
        """
        self.max_calls = max_calls_per_minute
        self.calls = deque()  # Timestamps of recent calls
        self.lock = threading.RLock()  # ✅ RLock for recursive calls
        
        # Statistics
        self.total_calls = 0
        self.total_wait_time = 0.0
    
    def acquire(self) -> float:
        """
        Wait if necessary, then allow call to proceed.
        
        Blocks calling thread if rate limit reached until capacity available.
        Uses recursive call pattern - RLock allows same thread to re-acquire.
        
        Returns:
            Wait time in seconds (0.0 if no wait needed)
        """
        with self.lock:
            now = time.time()
            wait_time = 0.0
            
            # Remove calls older than 60 seconds (sliding window)
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            
            # If at capacity, wait for oldest call to expire
            if len(self.calls) >= self.max_calls:
                oldest_call = self.calls[0]
                wait_time = 60 - (now - oldest_call) + 0.1  # +0.1s buffer
                
                # Release lock while sleeping (allow other threads)
                # RLock tracks acquisition count, so this is safe
                self.lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self.lock.acquire()
                
                # Recursive call - RLock allows this
                return wait_time + self.acquire()
            
            # Record this call
            self.calls.append(now)
            self.total_calls += 1
            self.total_wait_time += wait_time
            
            return wait_time
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with total_calls, total_wait_time, current_window_usage
        """
        with self.lock:
            now = time.time()
            # Count calls in current 60s window
            current_window = sum(1 for t in self.calls if now - t < 60)
            
            return {
                'total_calls': self.total_calls,
                'total_wait_time_sec': round(self.total_wait_time, 2),
                'current_window_usage': current_window,
                'max_capacity': self.max_calls,
                'utilization_pct': round(100 * current_window / self.max_calls, 1)
            }
    
    def reset_stats(self):
        """Reset statistics counters"""
        with self.lock:
            self.total_calls = 0
            self.total_wait_time = 0.0


# Example usage
if __name__ == "__main__":
    # Test rate limiter
    limiter = RateLimiter(max_calls_per_minute=10)  # Low limit for testing
    
    print("Testing rate limiter with 10 calls/min limit...")
    for i in range(15):
        start = time.time()
        wait = limiter.acquire()
        elapsed = time.time() - start
        
        if wait > 0:
            print(f"Call {i+1}: Waited {wait:.2f}s (throttled)")
        else:
            print(f"Call {i+1}: No wait (under limit)")
        
        time.sleep(0.1)  # Simulate work
    
    print("\nFinal stats:", limiter.get_stats())