"""
Unit tests for optimization scheduling and resource management system.
"""

import unittest
import numpy as np
import time
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from enhanced_timeseries.optimization.scheduler import (
    JobStatus, JobPriority, ResourceType, ResourceRequirements,
    OptimizationJob, SystemResources, ResourceMonitor, JobQueue,
    OptimizationScheduler, create_optimization_job, create_scheduler_with_defaults
)


class TestResourceRequirements(unittest.TestCase):
    """Test ResourceRequirements class."""
    
    def test_requirements_creation_and_serialization(self):
        """Test resource requirements creation and serialization."""
        requirements = ResourceRequirements(
            cpu_cores=4,
            memory_gb=8.0,
            gpu_count=1,
            gpu_memory_gb=4.0,
            disk_space_gb=10.0,
            estimated_duration_minutes=120
        )
        
        self.assertEqual(requirements.cpu_cores, 4)
        self.assertEqual(requirements.memory_gb, 8.0)
        self.assertEqual(requirements.gpu_count, 1)
        
        # Test serialization
        req_dict = requirements.to_dict()
        self.assertIsInstance(req_dict, dict)
        self.assertEqual(req_dict['cpu_cores'], 4)
        self.assertEqual(req_dict['gpu_count'], 1)


class TestOptimizationJob(unittest.TestCase):
    """Test OptimizationJob class."""
    
    def test_job_creation_and_serialization(self):
        """Test job creation and serialization."""
        def dummy_objective(params):
            return params.get('x', 0) ** 2
        
        requirements = ResourceRequirements(cpu_cores=2, memory_gb=4.0)
        
        job = OptimizationJob(
            job_id="test_job_1",
            name="Test Optimization Job",
            objective_function=dummy_objective,
            parameters=[],  # Would contain Parameter objects in real use
            priority=JobPriority.HIGH,
            resource_requirements=requirements,
            max_evaluations=25,
            timeout_minutes=60
        )
        
        self.assertEqual(job.job_id, "test_job_1")
        self.assertEqual(job.priority, JobPriority.HIGH)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.progress, 0.0)
        
        # Test serialization (excludes callable)
        job_dict = job.to_dict()
        self.assertIsInstance(job_dict, dict)
        self.assertEqual(job_dict['job_id'], "test_job_1")
        self.assertEqual(job_dict['priority'], JobPriority.HIGH.value)
        self.assertNotIn('objective_function', job_dict)  # Callable excluded


class TestSystemResources(unittest.TestCase):
    """Test SystemResources class."""
    
    def test_resources_creation_and_serialization(self):
        """Test system resources creation and serialization."""
        resources = SystemResources(
            cpu_cores=8,
            memory_gb=16.0,
            gpu_count=2,
            gpu_memory_gb=8.0,
            disk_space_gb=500.0,
            cpu_usage_percent=25.0,
            memory_usage_percent=60.0,
            gpu_usage_percent=10.0
        )
        
        self.assertEqual(resources.cpu_cores, 8)
        self.assertEqual(resources.memory_gb, 16.0)
        self.assertEqual(resources.cpu_usage_percent, 25.0)
        
        # Test serialization
        res_dict = resources.to_dict()
        self.assertIsInstance(res_dict, dict)
        self.assertEqual(res_dict['cpu_cores'], 8)
        self.assertEqual(res_dict['cpu_usage_percent'], 25.0)


class TestResourceMonitor(unittest.TestCase):
    """Test ResourceMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = ResourceMonitor(update_interval=1)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    def test_monitor_creation(self):
        """Test resource monitor creation."""
        self.assertIsInstance(self.monitor.resources, SystemResources)
        self.assertFalse(self.monitor.monitoring)
        self.assertIsNone(self.monitor.monitor_thread)
    
    def test_get_system_resources(self):
        """Test getting system resources."""
        resources = self.monitor._get_system_resources()
        
        self.assertIsInstance(resources, SystemResources)
        self.assertGreater(resources.cpu_cores, 0)
        self.assertGreater(resources.memory_gb, 0)
        self.assertGreaterEqual(resources.gpu_count, 0)
    
    def test_resource_allocation_check(self):
        """Test resource allocation checking."""
        # Test with reasonable requirements
        reasonable_req = ResourceRequirements(
            cpu_cores=1,
            memory_gb=1.0,
            gpu_count=0
        )
        
        can_allocate = self.monitor.can_allocate_resources(reasonable_req)
        self.assertTrue(can_allocate)
        
        # Test with excessive requirements
        excessive_req = ResourceRequirements(
            cpu_cores=1000,  # Unrealistic
            memory_gb=1000.0,
            gpu_count=0
        )
        
        can_allocate = self.monitor.can_allocate_resources(excessive_req)
        self.assertFalse(can_allocate)
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring."""
        self.assertFalse(self.monitor.monitoring)
        
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring)
        self.assertIsNotNone(self.monitor.monitor_thread)
        
        # Let it run briefly
        time.sleep(0.1)
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring)


class TestJobQueue(unittest.TestCase):
    """Test JobQueue class."""
    
    def setUp(self):
        """Set up test environment."""
        self.queue = JobQueue()
    
    def create_test_job(self, job_id: str, priority: JobPriority = JobPriority.NORMAL) -> OptimizationJob:
        """Create a test job."""
        def dummy_objective(params):
            return 0.0
        
        return OptimizationJob(
            job_id=job_id,
            name=f"Test Job {job_id}",
            objective_function=dummy_objective,
            parameters=[],
            priority=priority,
            resource_requirements=ResourceRequirements()
        )
    
    def test_queue_creation(self):
        """Test job queue creation."""
        self.assertIsInstance(self.queue.jobs, dict)
        self.assertEqual(len(self.queue.jobs), 0)
        self.assertEqual(len(self.queue.running_jobs), 0)
        self.assertEqual(len(self.queue.completed_jobs), 0)
    
    def test_add_and_get_job(self):
        """Test adding and getting jobs."""
        job = self.create_test_job("job_1", JobPriority.HIGH)
        
        self.queue.add_job(job)
        
        # Check job was added
        self.assertIn("job_1", self.queue.jobs)
        self.assertEqual(len(self.queue.jobs), 1)
        
        # Get next job
        next_job = self.queue.get_next_job()
        self.assertIsNotNone(next_job)
        self.assertEqual(next_job.job_id, "job_1")
    
    def test_job_priority_ordering(self):
        """Test job priority ordering."""
        # Add jobs with different priorities
        low_job = self.create_test_job("low_job", JobPriority.LOW)
        high_job = self.create_test_job("high_job", JobPriority.HIGH)
        normal_job = self.create_test_job("normal_job", JobPriority.NORMAL)
        
        # Add in random order
        self.queue.add_job(normal_job)
        self.queue.add_job(low_job)
        self.queue.add_job(high_job)
        
        # Should get high priority job first
        next_job = self.queue.get_next_job()
        self.assertEqual(next_job.job_id, "high_job")
        
        # Then normal priority
        next_job = self.queue.get_next_job()
        self.assertEqual(next_job.job_id, "normal_job")
        
        # Finally low priority
        next_job = self.queue.get_next_job()
        self.assertEqual(next_job.job_id, "low_job")
    
    def test_job_status_updates(self):
        """Test job status updates."""
        job = self.create_test_job("job_1")
        self.queue.add_job(job)
        
        # Update to running
        self.queue.update_job_status("job_1", JobStatus.RUNNING, progress=25.0)
        
        updated_job = self.queue.get_job("job_1")
        self.assertEqual(updated_job.status, JobStatus.RUNNING)
        self.assertEqual(updated_job.progress, 25.0)
        self.assertIsNotNone(updated_job.started_timestamp)
        self.assertIn("job_1", self.queue.running_jobs)
        
        # Update to completed
        result = {"best_value": 0.95}
        self.queue.update_job_status("job_1", JobStatus.COMPLETED, progress=100.0, result=result)
        
        updated_job = self.queue.get_job("job_1")
        self.assertEqual(updated_job.status, JobStatus.COMPLETED)
        self.assertEqual(updated_job.progress, 100.0)
        self.assertEqual(updated_job.result, result)
        self.assertIsNotNone(updated_job.completed_timestamp)
        self.assertNotIn("job_1", self.queue.running_jobs)
        self.assertIn("job_1", self.queue.completed_jobs)
    
    def test_cancel_job(self):
        """Test job cancellation."""
        job = self.create_test_job("job_1")
        self.queue.add_job(job)
        
        # Cancel pending job
        success = self.queue.cancel_job("job_1")
        self.assertTrue(success)
        
        cancelled_job = self.queue.get_job("job_1")
        self.assertEqual(cancelled_job.status, JobStatus.CANCELLED)
        self.assertIsNotNone(cancelled_job.completed_timestamp)
        
        # Try to cancel already cancelled job
        success = self.queue.cancel_job("job_1")
        self.assertFalse(success)
    
    def test_get_jobs_by_status(self):
        """Test getting jobs by status."""
        job1 = self.create_test_job("job_1")
        job2 = self.create_test_job("job_2")
        
        self.queue.add_job(job1)
        self.queue.add_job(job2)
        
        # Both should be pending
        pending_jobs = self.queue.get_jobs_by_status(JobStatus.PENDING)
        self.assertEqual(len(pending_jobs), 2)
        
        # Update one to running
        self.queue.update_job_status("job_1", JobStatus.RUNNING)
        
        pending_jobs = self.queue.get_jobs_by_status(JobStatus.PENDING)
        running_jobs = self.queue.get_jobs_by_status(JobStatus.RUNNING)
        
        self.assertEqual(len(pending_jobs), 1)
        self.assertEqual(len(running_jobs), 1)
    
    def test_queue_statistics(self):
        """Test queue statistics."""
        # Add jobs with different statuses
        job1 = self.create_test_job("job_1")
        job2 = self.create_test_job("job_2")
        job3 = self.create_test_job("job_3")
        
        self.queue.add_job(job1)
        self.queue.add_job(job2)
        self.queue.add_job(job3)
        
        # Update statuses
        self.queue.update_job_status("job_1", JobStatus.RUNNING)
        self.queue.update_job_status("job_2", JobStatus.COMPLETED)
        
        stats = self.queue.get_queue_statistics()
        
        self.assertEqual(stats['total_jobs'], 3)
        self.assertEqual(stats['status_counts']['PENDING'], 1)
        self.assertEqual(stats['status_counts']['RUNNING'], 1)
        self.assertEqual(stats['status_counts']['COMPLETED'], 1)
        self.assertEqual(stats['running_jobs'], 1)
        self.assertEqual(stats['completed_jobs'], 1)


class TestOptimizationScheduler(unittest.TestCase):
    """Test OptimizationScheduler class."""
    
    def setUp(self):
        """Set up test environment."""
        self.scheduler = OptimizationScheduler(
            max_concurrent_jobs=2,
            resource_safety_margin=0.1,
            job_timeout_minutes=5
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if self.scheduler.running:
            self.scheduler.stop()
    
    def create_test_job(self, job_id: str, duration: float = 0.1) -> OptimizationJob:
        """Create a test job with controllable duration."""
        def test_objective(params):
            time.sleep(duration)  # Simulate work
            return np.random.random()
        
        from enhanced_timeseries.optimization.bayesian_optimizer import Parameter, ParameterType
        
        parameters = [
            Parameter("x", ParameterType.CONTINUOUS, bounds=(0, 1))
        ]
        
        return OptimizationJob(
            job_id=job_id,
            name=f"Test Job {job_id}",
            objective_function=test_objective,
            parameters=parameters,
            priority=JobPriority.NORMAL,
            resource_requirements=ResourceRequirements(cpu_cores=1, memory_gb=0.5),
            max_evaluations=3  # Keep small for testing
        )
    
    def test_scheduler_creation(self):
        """Test scheduler creation."""
        self.assertEqual(self.scheduler.max_concurrent_jobs, 2)
        self.assertFalse(self.scheduler.running)
        self.assertIsNotNone(self.scheduler.resource_monitor)
        self.assertIsNotNone(self.scheduler.job_queue)
    
    def test_scheduler_start_stop(self):
        """Test scheduler start and stop."""
        self.assertFalse(self.scheduler.running)
        
        self.scheduler.start()
        self.assertTrue(self.scheduler.running)
        self.assertTrue(self.scheduler.resource_monitor.monitoring)
        self.assertIsNotNone(self.scheduler.executor)
        
        self.scheduler.stop()
        self.assertFalse(self.scheduler.running)
        self.assertFalse(self.scheduler.resource_monitor.monitoring)
    
    def test_submit_and_execute_job(self):
        """Test job submission and execution."""
        self.scheduler.start()
        
        job = self.create_test_job("test_job", duration=0.05)
        job_id = self.scheduler.submit_job(job)
        
        self.assertEqual(job_id, "test_job")
        
        # Wait for job to complete
        max_wait = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            job_status = self.scheduler.get_job_status(job_id)
            if job_status and job_status.status == JobStatus.COMPLETED:
                break
            time.sleep(0.1)
        
        # Check job completed successfully
        final_job = self.scheduler.get_job_status(job_id)
        self.assertIsNotNone(final_job)
        self.assertEqual(final_job.status, JobStatus.COMPLETED)
        self.assertIsNotNone(final_job.result)
        self.assertEqual(final_job.progress, 100.0)
    
    def test_concurrent_job_execution(self):
        """Test concurrent job execution."""
        self.scheduler.start()
        
        # Submit multiple jobs
        job_ids = []
        for i in range(3):
            job = self.create_test_job(f"job_{i}", duration=0.1)
            job_id = self.scheduler.submit_job(job)
            job_ids.append(job_id)
        
        # Wait for all jobs to complete
        max_wait = 15  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            completed_count = 0
            for job_id in job_ids:
                job_status = self.scheduler.get_job_status(job_id)
                if job_status and job_status.status == JobStatus.COMPLETED:
                    completed_count += 1
            
            if completed_count == len(job_ids):
                break
            
            time.sleep(0.1)
        
        # Check all jobs completed
        for job_id in job_ids:
            job_status = self.scheduler.get_job_status(job_id)
            self.assertIsNotNone(job_status)
            self.assertEqual(job_status.status, JobStatus.COMPLETED)
    
    def test_job_cancellation(self):
        """Test job cancellation."""
        self.scheduler.start()
        
        # Submit a long-running job
        job = self.create_test_job("long_job", duration=5.0)
        job_id = self.scheduler.submit_job(job)
        
        # Wait for job to start
        time.sleep(0.5)
        
        # Cancel the job
        success = self.scheduler.cancel_job(job_id)
        self.assertTrue(success)
        
        # Check job was cancelled
        job_status = self.scheduler.get_job_status(job_id)
        self.assertEqual(job_status.status, JobStatus.CANCELLED)
    
    def test_system_status(self):
        """Test getting system status."""
        status = self.scheduler.get_system_status()
        
        self.assertIn('scheduler_running', status)
        self.assertIn('max_concurrent_jobs', status)
        self.assertIn('system_resources', status)
        self.assertIn('queue_statistics', status)
        
        self.assertEqual(status['max_concurrent_jobs'], 2)
        self.assertIsInstance(status['system_resources'], dict)
        self.assertIsInstance(status['queue_statistics'], dict)
    
    def test_job_history(self):
        """Test getting job history."""
        self.scheduler.start()
        
        # Submit and complete a job
        job = self.create_test_job("history_job", duration=0.05)
        job_id = self.scheduler.submit_job(job)
        
        # Wait for completion
        time.sleep(2)
        
        # Get history
        history = self.scheduler.get_job_history(hours=1)
        
        self.assertGreater(len(history), 0)
        
        # Find our job in history
        our_job = next((j for j in history if j['job_id'] == job_id), None)
        self.assertIsNotNone(our_job)
        self.assertEqual(our_job['job_id'], job_id)
    
    def test_export_job_results(self):
        """Test exporting job results."""
        self.scheduler.start()
        
        # Submit and complete a job
        job = self.create_test_job("export_job", duration=0.05)
        job_id = self.scheduler.submit_job(job)
        
        # Wait for completion
        max_wait = 10
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            job_status = self.scheduler.get_job_status(job_id)
            if job_status and job_status.status == JobStatus.COMPLETED:
                break
            time.sleep(0.1)
        
        # Export results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            success = self.scheduler.export_job_results(job_id, temp_filepath)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(temp_filepath))
            
            # Check export content
            with open(temp_filepath, 'r') as f:
                export_data = json.load(f)
            
            self.assertIn('job_info', export_data)
            self.assertEqual(export_data['job_info']['job_id'], job_id)
            
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_optimization_job(self):
        """Test creating optimization job with utility function."""
        def test_objective(params):
            return params.get('x', 0) ** 2
        
        from enhanced_timeseries.optimization.bayesian_optimizer import Parameter, ParameterType
        
        parameters = [
            Parameter("x", ParameterType.CONTINUOUS, bounds=(0, 1))
        ]
        
        job = create_optimization_job(
            name="Test Optimization",
            objective_function=test_objective,
            parameters=parameters,
            priority=JobPriority.HIGH,
            cpu_cores=2,
            memory_gb=4.0,
            gpu_count=1,
            max_evaluations=25
        )
        
        self.assertIsInstance(job, OptimizationJob)
        self.assertEqual(job.name, "Test Optimization")
        self.assertEqual(job.priority, JobPriority.HIGH)
        self.assertEqual(job.resource_requirements.cpu_cores, 2)
        self.assertEqual(job.resource_requirements.memory_gb, 4.0)
        self.assertEqual(job.resource_requirements.gpu_count, 1)
        self.assertEqual(job.max_evaluations, 25)
    
    def test_create_scheduler_with_defaults(self):
        """Test creating scheduler with default settings."""
        scheduler = create_scheduler_with_defaults()
        
        self.assertIsInstance(scheduler, OptimizationScheduler)
        self.assertGreater(scheduler.max_concurrent_jobs, 0)
        self.assertEqual(scheduler.resource_safety_margin, 0.2)
        self.assertEqual(scheduler.job_timeout_minutes, 240)
        
        # Clean up
        if scheduler.running:
            scheduler.stop()


if __name__ == '__main__':
    unittest.main()