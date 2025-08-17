"""
Optimization scheduling and resource management system.
Handles job queuing, resource allocation, and parallel optimization runs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import threading
import time
import json
import queue
import multiprocessing
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Try to import GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.info("GPUtil not available - GPU monitoring disabled")


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ResourceType(Enum):
    """Resource types for allocation."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"


@dataclass
class ResourceRequirements:
    """Resource requirements for a job."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    disk_space_gb: float = 1.0
    estimated_duration_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': self.gpu_memory_gb,
            'disk_space_gb': self.disk_space_gb,
            'estimated_duration_minutes': self.estimated_duration_minutes
        }


@dataclass
class OptimizationJob:
    """Optimization job definition."""
    job_id: str
    name: str
    objective_function: Callable[[Dict[str, Any]], float]
    parameters: List[Any]  # Parameter definitions
    priority: JobPriority
    resource_requirements: ResourceRequirements
    max_evaluations: int = 50
    timeout_minutes: int = 120
    created_timestamp: datetime = field(default_factory=datetime.now)
    scheduled_timestamp: Optional[datetime] = None
    started_timestamp: Optional[datetime] = None
    completed_timestamp: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callable)."""
        return {
            'job_id': self.job_id,
            'name': self.name,
            'priority': self.priority.value,
            'resource_requirements': self.resource_requirements.to_dict(),
            'max_evaluations': self.max_evaluations,
            'timeout_minutes': self.timeout_minutes,
            'created_timestamp': self.created_timestamp.isoformat(),
            'scheduled_timestamp': self.scheduled_timestamp.isoformat() if self.scheduled_timestamp else None,
            'started_timestamp': self.started_timestamp.isoformat() if self.started_timestamp else None,
            'completed_timestamp': self.completed_timestamp.isoformat() if self.completed_timestamp else None,
            'status': self.status.value,
            'progress': self.progress,
            'result': self.result,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class SystemResources:
    """System resource information."""
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    disk_space_gb: float
    
    # Current usage
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': self.gpu_memory_gb,
            'disk_space_gb': self.disk_space_gb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'gpu_usage_percent': self.gpu_usage_percent,
            'gpu_memory_usage_percent': self.gpu_memory_usage_percent,
            'disk_usage_percent': self.disk_usage_percent
        }


class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, update_interval: int = 5):
        """
        Initialize resource monitor.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.resources = self._get_system_resources()
        self.monitoring = False
        self.monitor_thread = None
        
    def _get_system_resources(self) -> SystemResources:
        """Get current system resources."""
        # CPU and Memory
        cpu_cores = psutil.cpu_count(logical=True)
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.total / (1024**3)
        
        # Disk space
        disk_info = psutil.disk_usage('/')
        disk_space_gb = disk_info.total / (1024**3)
        
        # GPU information
        gpu_count = 0
        gpu_memory_gb = 0.0
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_count = len(gpus)
                gpu_memory_gb = sum(gpu.memoryTotal for gpu in gpus) / 1024  # Convert MB to GB
            except Exception as e:
                logger.warning(f"Failed to get GPU information: {e}")
        
        return SystemResources(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            disk_space_gb=disk_space_gb
        )
    
    def _update_resource_usage(self):
        """Update current resource usage."""
        # CPU and Memory usage
        self.resources.cpu_usage_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        self.resources.memory_usage_percent = memory_info.percent
        
        # Disk usage
        disk_info = psutil.disk_usage('/')
        self.resources.disk_usage_percent = (disk_info.used / disk_info.total) * 100
        
        # GPU usage
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.resources.gpu_usage_percent = np.mean([gpu.load * 100 for gpu in gpus])
                    self.resources.gpu_memory_usage_percent = np.mean([gpu.memoryUtil * 100 for gpu in gpus])
            except Exception as e:
                logger.warning(f"Failed to update GPU usage: {e}")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        while self.monitoring:
            try:
                self._update_resource_usage()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def get_current_resources(self) -> SystemResources:
        """Get current resource information."""
        self._update_resource_usage()
        return self.resources
    
    def can_allocate_resources(self, requirements: ResourceRequirements, 
                             safety_margin: float = 0.1) -> bool:
        """
        Check if resources can be allocated.
        
        Args:
            requirements: Resource requirements
            safety_margin: Safety margin (0.1 = 10% buffer)
            
        Returns:
            True if resources can be allocated
        """
        current = self.get_current_resources()
        
        # Check CPU
        available_cpu = current.cpu_cores * (1 - current.cpu_usage_percent / 100)
        if requirements.cpu_cores > available_cpu * (1 - safety_margin):
            return False
        
        # Check Memory
        available_memory = current.memory_gb * (1 - current.memory_usage_percent / 100)
        if requirements.memory_gb > available_memory * (1 - safety_margin):
            return False
        
        # Check GPU
        if requirements.gpu_count > 0:
            if current.gpu_count < requirements.gpu_count:
                return False
            
            available_gpu_memory = current.gpu_memory_gb * (1 - current.gpu_memory_usage_percent / 100)
            if requirements.gpu_memory_gb > available_gpu_memory * (1 - safety_margin):
                return False
        
        # Check Disk
        available_disk = current.disk_space_gb * (1 - current.disk_usage_percent / 100)
        if requirements.disk_space_gb > available_disk * (1 - safety_margin):
            return False
        
        return True


class JobQueue:
    """Priority-based job queue."""
    
    def __init__(self):
        """Initialize job queue."""
        self.jobs = {}
        self.pending_jobs = queue.PriorityQueue()
        self.running_jobs = {}
        self.completed_jobs = {}
        self.job_lock = threading.Lock()
    
    def add_job(self, job: OptimizationJob):
        """Add job to queue."""
        with self.job_lock:
            self.jobs[job.job_id] = job
            
            # Add to pending queue with priority (higher priority = lower number for PriorityQueue)
            priority_value = -job.priority.value  # Negative for correct ordering
            self.pending_jobs.put((priority_value, job.created_timestamp, job.job_id))
            
            logger.info(f"Added job {job.job_id} to queue with priority {job.priority.name}")
    
    def get_next_job(self) -> Optional[OptimizationJob]:
        """Get next job from queue."""
        try:
            _, _, job_id = self.pending_jobs.get_nowait()
            
            with self.job_lock:
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    if job.status == JobStatus.PENDING:
                        return job
            
            # Job was cancelled or modified, try next
            return self.get_next_job()
            
        except queue.Empty:
            return None
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         progress: float = None, result: Dict[str, Any] = None,
                         error_message: str = None):
        """Update job status."""
        with self.job_lock:
            if job_id not in self.jobs:
                return
            
            job = self.jobs[job_id]
            old_status = job.status
            job.status = status
            
            if progress is not None:
                job.progress = progress
            
            if result is not None:
                job.result = result
            
            if error_message is not None:
                job.error_message = error_message
            
            # Update timestamps
            if status == JobStatus.RUNNING and old_status == JobStatus.PENDING:
                job.started_timestamp = datetime.now()
                self.running_jobs[job_id] = job
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_timestamp = datetime.now()
                if job_id in self.running_jobs:
                    del self.running_jobs[job_id]
                self.completed_jobs[job_id] = job
    
    def get_job(self, job_id: str) -> Optional[OptimizationJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def get_jobs_by_status(self, status: JobStatus) -> List[OptimizationJob]:
        """Get jobs by status."""
        with self.job_lock:
            return [job for job in self.jobs.values() if job.status == status]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self.job_lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            job.status = JobStatus.CANCELLED
            job.completed_timestamp = datetime.now()
            
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            self.completed_jobs[job_id] = job
            
            logger.info(f"Cancelled job {job_id}")
            return True
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.job_lock:
            status_counts = {}
            for status in JobStatus:
                status_counts[status.name] = len(self.get_jobs_by_status(status))
            
            return {
                'total_jobs': len(self.jobs),
                'status_counts': status_counts,
                'pending_queue_size': self.pending_jobs.qsize(),
                'running_jobs': len(self.running_jobs),
                'completed_jobs': len(self.completed_jobs)
            }


class OptimizationScheduler:
    """Main optimization scheduler with resource management."""
    
    def __init__(self, 
                 max_concurrent_jobs: int = None,
                 resource_safety_margin: float = 0.1,
                 job_timeout_minutes: int = 180):
        """
        Initialize optimization scheduler.
        
        Args:
            max_concurrent_jobs: Maximum concurrent jobs (default: CPU cores)
            resource_safety_margin: Safety margin for resource allocation
            job_timeout_minutes: Default job timeout
        """
        self.max_concurrent_jobs = max_concurrent_jobs or multiprocessing.cpu_count()
        self.resource_safety_margin = resource_safety_margin
        self.job_timeout_minutes = job_timeout_minutes
        
        # Initialize components
        self.resource_monitor = ResourceMonitor()
        self.job_queue = JobQueue()
        
        # Scheduler state
        self.running = False
        self.scheduler_thread = None
        self.executor = None
        
        # Job execution tracking
        self.job_futures = {}
        
        logger.info(f"Initialized scheduler with max {self.max_concurrent_jobs} concurrent jobs")
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            return
        
        self.running = True
        self.resource_monitor.start_monitoring()
        
        # Initialize thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_jobs)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Optimization scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Wait for scheduler thread
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        logger.info("Optimization scheduler stopped")
    
    def submit_job(self, job: OptimizationJob) -> str:
        """
        Submit an optimization job.
        
        Args:
            job: Optimization job to submit
            
        Returns:
            Job ID
        """
        # Validate resource requirements
        if not self.resource_monitor.can_allocate_resources(
            job.resource_requirements, self.resource_safety_margin
        ):
            logger.warning(f"Insufficient resources for job {job.job_id}")
            job.status = JobStatus.FAILED
            job.error_message = "Insufficient system resources"
            return job.job_id
        
        # Add to queue
        self.job_queue.add_job(job)
        
        return job.job_id
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                self._process_pending_jobs()
                self._check_running_jobs()
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)
    
    def _process_pending_jobs(self):
        """Process pending jobs."""
        # Check if we can start more jobs
        running_count = len(self.job_queue.running_jobs)
        
        if running_count >= self.max_concurrent_jobs:
            return
        
        # Get next job
        job = self.job_queue.get_next_job()
        if not job:
            return
        
        # Check resources again
        if not self.resource_monitor.can_allocate_resources(
            job.resource_requirements, self.resource_safety_margin
        ):
            # Put job back in queue (it will be retried later)
            self.job_queue.pending_jobs.put((-job.priority.value, job.created_timestamp, job.job_id))
            return
        
        # Start job
        self._start_job(job)
    
    def _start_job(self, job: OptimizationJob):
        """Start executing a job."""
        logger.info(f"Starting job {job.job_id}: {job.name}")
        
        # Update job status
        self.job_queue.update_job_status(job.job_id, JobStatus.RUNNING)
        
        # Submit to executor
        future = self.executor.submit(self._execute_job, job)
        self.job_futures[job.job_id] = future
    
    def _execute_job(self, job: OptimizationJob) -> Dict[str, Any]:
        """Execute an optimization job."""
        try:
            from enhanced_timeseries.optimization.bayesian_optimizer import HyperparameterOptimizer
            
            # Create optimizer
            optimizer = HyperparameterOptimizer(
                parameters=job.parameters,
                objective_function=job.objective_function,
                max_evaluations=job.max_evaluations
            )
            
            # Run optimization with progress tracking
            def progress_callback(evaluation: int, total: int):
                progress = (evaluation / total) * 100
                self.job_queue.update_job_status(job.job_id, JobStatus.RUNNING, progress=progress)
            
            # Execute optimization
            best_params, best_value = optimizer.optimize(verbose=False)
            
            # Prepare result
            result = {
                'best_parameters': best_params,
                'best_objective_value': best_value,
                'optimization_history': optimizer.get_optimization_history(),
                'total_evaluations': optimizer.n_evaluations
            }
            
            # Update job status
            self.job_queue.update_job_status(
                job.job_id, JobStatus.COMPLETED, progress=100.0, result=result
            )
            
            logger.info(f"Completed job {job.job_id} with best value: {best_value:.6f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Job execution failed: {str(e)}"
            logger.error(f"Job {job.job_id} failed: {error_msg}")
            
            self.job_queue.update_job_status(
                job.job_id, JobStatus.FAILED, error_message=error_msg
            )
            
            raise e
    
    def _check_running_jobs(self):
        """Check status of running jobs."""
        completed_jobs = []
        
        for job_id, future in self.job_futures.items():
            if future.done():
                completed_jobs.append(job_id)
                
                try:
                    result = future.result()
                except Exception as e:
                    # Job already marked as failed in _execute_job
                    pass
        
        # Clean up completed jobs
        for job_id in completed_jobs:
            del self.job_futures[job_id]
    
    def get_job_status(self, job_id: str) -> Optional[OptimizationJob]:
        """Get job status."""
        return self.job_queue.get_job(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        # Cancel in queue
        success = self.job_queue.cancel_job(job_id)
        
        # Cancel running job if exists
        if job_id in self.job_futures:
            future = self.job_futures[job_id]
            future.cancel()
            del self.job_futures[job_id]
        
        return success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        resources = self.resource_monitor.get_current_resources()
        queue_stats = self.job_queue.get_queue_statistics()
        
        return {
            'scheduler_running': self.running,
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'system_resources': resources.to_dict(),
            'queue_statistics': queue_stats,
            'active_futures': len(self.job_futures)
        }
    
    def get_job_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get job history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for job in self.job_queue.jobs.values():
            if job.created_timestamp >= cutoff_time:
                history.append(job.to_dict())
        
        # Sort by creation time (newest first)
        history.sort(key=lambda x: x['created_timestamp'], reverse=True)
        
        return history
    
    def export_job_results(self, job_id: str, filepath: str) -> bool:
        """Export job results to file."""
        job = self.job_queue.get_job(job_id)
        if not job or not job.result:
            return False
        
        try:
            export_data = {
                'job_info': job.to_dict(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported job {job_id} results to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export job results: {e}")
            return False


# Utility functions
def create_optimization_job(name: str,
                          objective_function: Callable[[Dict[str, Any]], float],
                          parameters: List[Any],
                          priority: JobPriority = JobPriority.NORMAL,
                          cpu_cores: int = 1,
                          memory_gb: float = 1.0,
                          gpu_count: int = 0,
                          max_evaluations: int = 50) -> OptimizationJob:
    """
    Create an optimization job with default settings.
    
    Args:
        name: Job name
        objective_function: Objective function to optimize
        parameters: Parameter definitions
        priority: Job priority
        cpu_cores: Required CPU cores
        memory_gb: Required memory in GB
        gpu_count: Required GPU count
        max_evaluations: Maximum evaluations
        
    Returns:
        OptimizationJob instance
    """
    job_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    
    resource_requirements = ResourceRequirements(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_count * 4.0 if gpu_count > 0 else 0.0,  # Assume 4GB per GPU
        estimated_duration_minutes=max_evaluations * 2  # Rough estimate
    )
    
    return OptimizationJob(
        job_id=job_id,
        name=name,
        objective_function=objective_function,
        parameters=parameters,
        priority=priority,
        resource_requirements=resource_requirements,
        max_evaluations=max_evaluations
    )


def create_scheduler_with_defaults() -> OptimizationScheduler:
    """Create scheduler with default settings."""
    return OptimizationScheduler(
        max_concurrent_jobs=max(1, multiprocessing.cpu_count() // 2),
        resource_safety_margin=0.2,  # 20% safety margin
        job_timeout_minutes=240  # 4 hours
    )