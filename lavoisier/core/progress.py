"""
Progress tracking and reporting for Lavoisier pipelines.

This module provides utilities for tracking progress of long-running operations
and estimating remaining time.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
import threading
from collections import deque
import json

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track progress of operations with time estimation
    """
    def __init__(
        self, 
        total_steps: int = 100, 
        name: str = "Operation", 
        history_size: int = 10,
        callback: Optional[Callable[[float, str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize progress tracker
        
        Args:
            total_steps: Total number of steps (percent is calculated based on this)
            name: Name of the operation for display purposes
            history_size: Number of progress updates to store for time estimation
            callback: Optional callback to be called on progress updates
        """
        self.total_steps = max(1, total_steps)
        self.current_step = 0
        self.name = name
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.end_time: Optional[float] = None
        self.history: deque = deque(maxlen=history_size)
        self.callback = callback
        self.messages: List[str] = []
        self.details: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.is_paused = False
        self.pause_start_time: Optional[float] = None
        self.total_pause_time = 0.0
        
        # Initial progress point
        self._record_progress_point(0)
    
    def update(
        self, 
        current_step: Optional[int] = None, 
        increment: int = 0, 
        message: Optional[str] = None,
        **details
    ) -> Dict[str, Any]:
        """
        Update progress
        
        Args:
            current_step: Current step (if None, uses increment instead)
            increment: Number of steps to increment (ignored if current_step is provided)
            message: Optional status message
            **details: Additional details to store with the progress update
            
        Returns:
            Dict containing current progress information
        """
        with self.lock:
            if self.is_paused:
                return self.get_status()
                
            # Update step
            if current_step is not None:
                self.current_step = min(current_step, self.total_steps)
            else:
                self.current_step = min(self.current_step + increment, self.total_steps)
            
            # Record time of update
            current_time = time.time()
            self.last_update_time = current_time
            
            # Record progress point for time estimation
            self._record_progress_point(self.current_step)
            
            # Store message if provided
            if message:
                self.messages.append(message)
                
            # Update details
            self.details.update(details)
            
            # Call callback if provided
            status = self.get_status()
            if self.callback:
                try:
                    self.callback(
                        self.get_percent(), 
                        message or self.get_latest_message(), 
                        status
                    )
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")
                    
            return status
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status including estimated time remaining"""
        with self.lock:
            percent = self.get_percent()
            elapsed = self.get_elapsed_time()
            remaining = self.estimate_remaining_time()
            
            status = {
                "name": self.name,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "percent": percent,
                "elapsed_seconds": elapsed,
                "elapsed_formatted": self.format_time(elapsed),
                "is_complete": self.current_step >= self.total_steps,
                "is_paused": self.is_paused,
            }
            
            if remaining is not None:
                status["remaining_seconds"] = remaining
                status["remaining_formatted"] = self.format_time(remaining)
                status["eta"] = self.format_datetime(time.time() + remaining)
            
            if self.messages:
                status["latest_message"] = self.messages[-1]
                
            if self.details:
                status["details"] = self.details
                
            return status
    
    def get_percent(self) -> float:
        """Get progress as percentage"""
        with self.lock:
            return 100.0 * self.current_step / self.total_steps
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds, excluding paused time"""
        with self.lock:
            if self.end_time:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.time() - self.start_time
                
            # Subtract paused time
            elapsed -= self.total_pause_time
            
            # If currently paused, also subtract current pause time
            if self.is_paused and self.pause_start_time:
                elapsed -= (time.time() - self.pause_start_time)
                
            return max(0.0, elapsed)
    
    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time in seconds based on progress history"""
        with self.lock:
            if self.is_paused or self.current_step >= self.total_steps:
                return 0.0
                
            if not self.history or len(self.history) < 2:
                # Not enough history for estimation
                return None
                
            # Use recent history to estimate speed
            recent = list(self.history)[-min(len(self.history), 5):]
            
            steps_per_second_sum = 0.0
            weight_sum = 0.0
            
            # Calculate weighted average of steps per second
            for i, (tstamp, step) in enumerate(recent[1:], 1):
                prev_tstamp, prev_step = recent[i-1]
                
                # Calculate time difference, excluding pauses
                time_diff = tstamp - prev_tstamp
                
                if time_diff <= 0:
                    continue
                    
                step_diff = step - prev_step
                if step_diff <= 0:
                    continue
                
                # More weight to recent updates
                weight = i
                steps_per_second = step_diff / time_diff
                steps_per_second_sum += steps_per_second * weight
                weight_sum += weight
            
            if weight_sum <= 0:
                return None
                
            # Calculate weighted average
            avg_steps_per_second = steps_per_second_sum / weight_sum
            
            if avg_steps_per_second <= 0:
                return None
                
            # Calculate remaining time
            remaining_steps = self.total_steps - self.current_step
            remaining_time = remaining_steps / avg_steps_per_second
            
            return max(0.0, remaining_time)
    
    def complete(self, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark progress as complete
        
        Args:
            message: Optional completion message
            
        Returns:
            Final status dictionary
        """
        with self.lock:
            self.current_step = self.total_steps
            self.end_time = time.time()
            
            if message:
                self.messages.append(message)
                
            status = self.get_status()
            
            if self.callback:
                try:
                    self.callback(100.0, message or self.get_latest_message(), status)
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")
                    
            return status
    
    def pause(self, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Pause progress tracking
        
        Args:
            message: Optional pause message
            
        Returns:
            Current status dictionary
        """
        with self.lock:
            if not self.is_paused:
                self.is_paused = True
                self.pause_start_time = time.time()
                
                if message:
                    self.messages.append(message)
                    
                status = self.get_status()
                
                if self.callback:
                    try:
                        self.callback(self.get_percent(), message or self.get_latest_message(), status)
                    except Exception as e:
                        logger.error(f"Error in progress callback: {str(e)}")
                        
                return status
            
            return self.get_status()
    
    def resume(self, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume progress tracking
        
        Args:
            message: Optional resume message
            
        Returns:
            Current status dictionary
        """
        with self.lock:
            if self.is_paused:
                # Calculate the time spent paused
                if self.pause_start_time:
                    self.total_pause_time += time.time() - self.pause_start_time
                
                self.is_paused = False
                self.pause_start_time = None
                
                if message:
                    self.messages.append(message)
                    
                status = self.get_status()
                
                if self.callback:
                    try:
                        self.callback(self.get_percent(), message or self.get_latest_message(), status)
                    except Exception as e:
                        logger.error(f"Error in progress callback: {str(e)}")
                        
                return status
            
            return self.get_status()
    
    def get_latest_message(self) -> str:
        """Get the most recent status message"""
        with self.lock:
            if not self.messages:
                return f"{self.name}: {self.get_percent():.1f}%"
            return self.messages[-1]
    
    def _record_progress_point(self, step: int) -> None:
        """Record a progress point for time estimation"""
        with self.lock:
            current_time = time.time()
            self.history.append((current_time, step))
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in seconds to human readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            seconds = int(seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def format_datetime(timestamp: float) -> str:
        """Format a timestamp to a human readable datetime string"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def to_json(self) -> str:
        """Convert progress status to JSON string"""
        return json.dumps(self.get_status())


class MultiProgressTracker:
    """
    Track progress of multiple operations simultaneously
    """
    def __init__(self, callback: Optional[Callable[[str, float, str, Dict[str, Any]], None]] = None):
        """
        Initialize multi progress tracker
        
        Args:
            callback: Optional callback called on any progress update with (tracker_id, percent, message, status)
        """
        self.trackers: Dict[str, ProgressTracker] = {}
        self.callback = callback
        self.lock = threading.RLock()
    
    def create_tracker(
        self, 
        tracker_id: str, 
        total_steps: int = 100, 
        name: str = "Operation", 
        history_size: int = 10
    ) -> ProgressTracker:
        """
        Create a new progress tracker
        
        Args:
            tracker_id: Unique identifier for this tracker
            total_steps: Total number of steps
            name: Name of the operation
            history_size: History size for time estimation
            
        Returns:
            Created ProgressTracker instance
        """
        with self.lock:
            # Define a callback that calls the multi tracker callback
            def single_callback(percent: float, message: str, status: Dict[str, Any]) -> None:
                if self.callback:
                    self.callback(tracker_id, percent, message, status)
            
            tracker = ProgressTracker(
                total_steps=total_steps,
                name=name,
                history_size=history_size,
                callback=single_callback
            )
            
            self.trackers[tracker_id] = tracker
            return tracker
    
    def get_tracker(self, tracker_id: str) -> Optional[ProgressTracker]:
        """Get a progress tracker by ID"""
        with self.lock:
            return self.trackers.get(tracker_id)
    
    def update(
        self, 
        tracker_id: str, 
        current_step: Optional[int] = None, 
        increment: int = 0, 
        message: Optional[str] = None,
        **details
    ) -> Optional[Dict[str, Any]]:
        """
        Update a specific tracker
        
        Args:
            tracker_id: Tracker ID to update
            current_step: Current step
            increment: Increment amount
            message: Optional status message
            **details: Additional details
            
        Returns:
            Updated status or None if tracker not found
        """
        with self.lock:
            tracker = self.get_tracker(tracker_id)
            if tracker:
                return tracker.update(current_step, increment, message, **details)
            return None
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all trackers"""
        with self.lock:
            return {
                tracker_id: tracker.get_status()
                for tracker_id, tracker in self.trackers.items()
            }
    
    def get_overall_progress(self) -> float:
        """Get average progress across all trackers"""
        with self.lock:
            if not self.trackers:
                return 0.0
                
            total_percent = sum(tracker.get_percent() for tracker in self.trackers.values())
            return total_percent / len(self.trackers)
    
    def complete_all(self, message: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Complete all trackers"""
        with self.lock:
            results = {}
            for tracker_id, tracker in self.trackers.items():
                results[tracker_id] = tracker.complete(message)
            return results
    
    def to_json(self) -> str:
        """Convert all tracker statuses to JSON string"""
        return json.dumps(self.get_all_status())


# Singleton instance for application-wide progress tracking
global_progress = MultiProgressTracker()


def get_progress_tracker() -> MultiProgressTracker:
    """Get the global progress tracker instance"""
    global global_progress
    return global_progress 