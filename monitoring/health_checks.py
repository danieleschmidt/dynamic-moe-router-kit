"""Health check endpoints and system monitoring for dynamic MoE router."""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


class BaseHealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout = timeout_seconds
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._perform_check(), 
                timeout=self.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get('status', HealthStatus.UNKNOWN),
                message=result.get('message', 'Check completed'),
                duration_ms=duration_ms,
                details=result.get('details')
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {self.timeout}s",
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement the actual check."""
        raise NotImplementedError


class MemoryHealthCheck(BaseHealthCheck):
    """Check system memory usage."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        super().__init__("memory")
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100.0
            
            if usage_percent >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {usage_percent:.1%}"
            elif usage_percent >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {usage_percent:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1%}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'usage_percent': usage_percent,
                    'available_bytes': memory.available,
                    'total_bytes': memory.total,
                    'used_bytes': memory.used
                }
            }
            
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'psutil not available for memory monitoring'
            }


class DiskSpaceHealthCheck(BaseHealthCheck):
    """Check available disk space."""
    
    def __init__(self, path: str = "/", warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        super().__init__("disk_space")
        self.path = path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(self.path)
            usage_percent = used / total
            
            if usage_percent >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {usage_percent:.1%}"
            elif usage_percent >= self.warning_threshold:
                status = HealthStatus.DEGRADED  
                message = f"High disk usage: {usage_percent:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1%}"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'path': self.path,
                    'usage_percent': usage_percent,
                    'total_bytes': total,
                    'used_bytes': used,
                    'free_bytes': free
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Disk check failed: {str(e)}'
            }


class ModelHealthCheck(BaseHealthCheck):
    """Check model inference health."""
    
    def __init__(self, model_name: str = "dynamic-moe-router"):
        super().__init__(f"model_{model_name}")
        self.model_name = model_name
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            # Mock model health check - replace with actual model testing
            import numpy as np
            
            # Simulate model inference test
            test_input = np.random.randn(1, 10, 768)
            
            # This would be replaced with actual model inference
            inference_time = 0.05  # Mock timing
            
            if inference_time > 1.0:
                status = HealthStatus.UNHEALTHY
                message = f"Model inference too slow: {inference_time:.3f}s"
            elif inference_time > 0.5:
                status = HealthStatus.DEGRADED
                message = f"Model inference degraded: {inference_time:.3f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Model inference healthy: {inference_time:.3f}s"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'model_name': self.model_name,
                    'inference_time_seconds': inference_time,
                    'test_input_shape': test_input.shape
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Model health check failed: {str(e)}'
            }


class DatabaseHealthCheck(BaseHealthCheck):
    """Check database connectivity (if applicable)."""
    
    def __init__(self, connection_string: Optional[str] = None):
        super().__init__("database")
        self.connection_string = connection_string
    
    async def _perform_check(self) -> Dict[str, Any]:
        if not self.connection_string:
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'No database configured'
            }
        
        try:
            # Mock database connection check
            # Replace with actual database connectivity test
            connection_time = 0.01  # Mock timing
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': f'Database connected in {connection_time:.3f}s',
                'details': {
                    'connection_time_seconds': connection_time
                }
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Database connection failed: {str(e)}'
            }


class ExternalServiceHealthCheck(BaseHealthCheck):
    """Check external service availability."""
    
    def __init__(self, service_name: str, endpoint: str):
        super().__init__(f"external_{service_name}")
        self.service_name = service_name
        self.endpoint = endpoint
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(self.endpoint) as response:
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = f"{self.service_name} service available"
                    else:
                        status = HealthStatus.DEGRADED
                        message = f"{self.service_name} returned {response.status}"
                    
                    return {
                        'status': status,
                        'message': message,
                        'details': {
                            'service_name': self.service_name,
                            'endpoint': self.endpoint,
                            'status_code': response.status
                        }
                    }
                    
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'aiohttp not available for external service checks'
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'{self.service_name} service unavailable: {str(e)}'
            }


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self):
        self.checks: List[BaseHealthCheck] = []
        self.last_results: Dict[str, HealthCheckResult] = {}
    
    def add_check(self, health_check: BaseHealthCheck):
        """Add a health check to the monitor."""
        self.checks.append(health_check)
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks concurrently."""
        tasks = [check.check() for check in self.checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        check_results = {}
        for i, result in enumerate(results):
            check_name = self.checks[i].name
            
            if isinstance(result, Exception):
                check_results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed with exception: {str(result)}",
                    duration_ms=0.0
                )
            else:
                check_results[check_name] = result
        
        self.last_results = check_results
        return check_results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = await self.run_all_checks()
        
        # Determine overall status
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate summary statistics
        total_checks = len(results)
        healthy_checks = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        degraded_checks = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        unhealthy_checks = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        
        return {
            'status': overall_status.value,
            'timestamp': time.time(),
            'summary': {
                'total_checks': total_checks,
                'healthy': healthy_checks,
                'degraded': degraded_checks,
                'unhealthy': unhealthy_checks
            },
            'checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'duration_ms': result.duration_ms,
                    'details': result.details
                }
                for name, result in results.items()
            }
        }
    
    def get_health_json(self) -> str:
        """Get health status as JSON string."""
        # Use cached results for performance
        overall_status = HealthStatus.HEALTHY
        if any(r.status == HealthStatus.UNHEALTHY for r in self.last_results.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in self.last_results.values()):
            overall_status = HealthStatus.DEGRADED
        
        health_data = {
            'status': overall_status.value,
            'timestamp': time.time(),
            'checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'duration_ms': result.duration_ms
                }
                for name, result in self.last_results.items()
            }
        }
        
        return json.dumps(health_data, indent=2)


# Example setup function
def setup_default_health_monitor() -> HealthMonitor:
    """Set up health monitor with default checks."""
    monitor = HealthMonitor()
    
    # Add standard health checks
    monitor.add_check(MemoryHealthCheck())
    monitor.add_check(DiskSpaceHealthCheck())
    monitor.add_check(ModelHealthCheck())
    monitor.add_check(DatabaseHealthCheck())
    
    return monitor


# Example usage
if __name__ == "__main__":
    async def main():
        monitor = setup_default_health_monitor()
        
        # Run health checks
        health_status = await monitor.get_overall_health() 
        
        print("Health Status:")
        print(json.dumps(health_status, indent=2))
        
        # Simulate continuous monitoring
        for i in range(5):
            await asyncio.sleep(2)
            health_status = await monitor.get_overall_health()
            print(f"\nHealth check {i+1}:")
            print(f"Overall status: {health_status['status']}")
            print(f"Healthy checks: {health_status['summary']['healthy']}/{health_status['summary']['total_checks']}")
    
    asyncio.run(main())