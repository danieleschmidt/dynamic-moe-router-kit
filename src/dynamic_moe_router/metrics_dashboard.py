"""Real-time metrics dashboard for dynamic MoE routing."""

import json
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """A snapshot of router metrics at a specific time."""
    timestamp: float
    request_count: int
    avg_experts_per_token: float
    flop_reduction: float
    processing_time_ms: float
    error_rate: float
    expert_utilization: List[float]
    router_health: str


class MetricsCollector:
    """Collects and aggregates metrics from router operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics_history = deque(maxlen=max_history)
        self._error_count = 0
        self._total_requests = 0
        self._lock = threading.Lock()
        
        # Per-minute aggregations
        self._minute_buckets = defaultdict(list)
        
    def record_request(self, 
                      result: Dict[str, Any], 
                      processing_time_ms: float,
                      error: bool = False) -> None:
        """Record a single request's metrics."""
        with self._lock:
            self._total_requests += 1
            if error:
                self._error_count += 1
            
            routing_info = result.get('routing_info', {})
            production_info = result.get('production_info', {})
            
            snapshot = MetricSnapshot(
                timestamp=time.time(),
                request_count=self._total_requests,
                avg_experts_per_token=routing_info.get('avg_experts_per_token', 0.0),
                flop_reduction=routing_info.get('flop_reduction', 0.0),
                processing_time_ms=processing_time_ms,
                error_rate=self._error_count / self._total_requests,
                expert_utilization=routing_info.get('expert_utilization', []),
                router_health=production_info.get('router_health', 'unknown')
            )
            
            self._metrics_history.append(snapshot)
            
            # Add to minute bucket for aggregation
            minute_key = int(snapshot.timestamp // 60)
            self._minute_buckets[minute_key].append(snapshot)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current aggregated metrics."""
        with self._lock:
            if not self._metrics_history:
                return {}
            
            recent_snapshots = list(self._metrics_history)[-100:]  # Last 100 requests
            
            if not recent_snapshots:
                return {}
            
            # Calculate aggregations
            avg_processing_time = sum(s.processing_time_ms for s in recent_snapshots) / len(recent_snapshots)
            avg_experts = sum(s.avg_experts_per_token for s in recent_snapshots) / len(recent_snapshots)
            avg_flop_reduction = sum(s.flop_reduction for s in recent_snapshots) / len(recent_snapshots)
            
            # Calculate throughput (requests per second)
            if len(recent_snapshots) > 1:
                time_span = recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp
                throughput = len(recent_snapshots) / max(time_span, 1.0)
            else:
                throughput = 0.0
            
            latest = recent_snapshots[-1]
            
            return {
                'total_requests': self._total_requests,
                'error_count': self._error_count,
                'error_rate': latest.error_rate,
                'throughput_rps': throughput,
                'avg_processing_time_ms': avg_processing_time,
                'avg_experts_per_token': avg_experts,
                'avg_flop_reduction': avg_flop_reduction,
                'expert_utilization': latest.expert_utilization,
                'router_health': latest.router_health,
                'timestamp': latest.timestamp
            }
    
    def get_time_series(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get time series data for the last N minutes."""
        with self._lock:
            cutoff_time = time.time() - (minutes * 60)
            
            time_series = []
            for snapshot in self._metrics_history:
                if snapshot.timestamp >= cutoff_time:
                    time_series.append({
                        'timestamp': snapshot.timestamp,
                        'processing_time_ms': snapshot.processing_time_ms,
                        'avg_experts_per_token': snapshot.avg_experts_per_token,
                        'flop_reduction': snapshot.flop_reduction,
                        'error_rate': snapshot.error_rate
                    })
            
            return time_series


class RealtimeDashboard:
    """Real-time dashboard for monitoring MoE router performance."""
    
    def __init__(self, router, update_interval: float = 1.0):
        self.router = router
        self.update_interval = update_interval
        self.metrics_collector = MetricsCollector()
        self._running = False
        self._dashboard_thread = None
        
    def start(self) -> None:
        """Start the real-time dashboard."""
        if self._running:
            return
            
        self._running = True
        self._dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self._dashboard_thread.start()
        
        logger.info("Real-time dashboard started")
    
    def stop(self) -> None:
        """Stop the real-time dashboard."""
        self._running = False
        if self._dashboard_thread:
            self._dashboard_thread.join()
        
        logger.info("Real-time dashboard stopped")
    
    def _dashboard_loop(self) -> None:
        """Main dashboard update loop."""
        while self._running:
            try:
                self._update_display()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
    
    def _update_display(self) -> None:
        """Update the dashboard display."""
        # Clear screen
        print("\033[2J\033[H", end='')
        
        # Header
        print("🔥 Dynamic MoE Router - Real-time Dashboard")
        print("=" * 80)
        print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get current metrics
        metrics = self.metrics_collector.get_current_metrics()
        
        if not metrics:
            print("⏳ Waiting for data...")
            return
        
        # Performance metrics
        print("📊 Performance Metrics:")
        print("-" * 40)
        print(f"📈 Total Requests: {metrics['total_requests']}")
        print(f"⚡ Throughput: {metrics['throughput_rps']:.2f} req/sec")
        print(f"⏱️  Avg Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
        print(f"❌ Error Rate: {metrics['error_rate']:.3f} ({metrics['error_count']} errors)")
        print(f"🏥 Router Health: {metrics['router_health']}")
        print()
        
        # Routing efficiency
        print("🎯 Routing Efficiency:")
        print("-" * 40)
        print(f"👥 Avg Experts per Token: {metrics['avg_experts_per_token']:.2f}")
        print(f"💾 FLOP Reduction: {metrics['avg_flop_reduction']*100:.1f}%")
        print()
        
        # Expert utilization
        if metrics['expert_utilization']:
            print("🤖 Expert Utilization:")
            print("-" * 40)
            for i, util in enumerate(metrics['expert_utilization'][:8]):  # Show first 8
                bar_length = int(util * 20)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"Expert {i:2d}: {bar} {util:.3f}")
            print()
        
        # Recent activity (mini time series)
        time_series = self.metrics_collector.get_time_series(minutes=1)
        if len(time_series) > 5:
            print("📈 Recent Activity (last minute):")
            print("-" * 40)
            
            # Simple ASCII chart for processing time
            values = [point['processing_time_ms'] for point in time_series[-20:]]
            if values:
                max_val = max(values)
                min_val = min(values)
                if max_val > min_val:
                    normalized = [(v - min_val) / (max_val - min_val) for v in values]
                    chart_line = ""
                    for norm_val in normalized:
                        height = int(norm_val * 8)
                        chart_line += ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"][height]
                    print(f"Processing Time: {chart_line}")
                    print(f"Range: {min_val:.1f}ms - {max_val:.1f}ms")
        
        print()
        print("Press Ctrl+C to stop dashboard")
    
    def record_request(self, result: Dict[str, Any], processing_time_ms: float, error: bool = False) -> None:
        """Record a request for dashboard metrics."""
        self.metrics_collector.record_request(result, processing_time_ms, error)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return self.metrics_collector.get_current_metrics()
    
    def export_metrics(self, filename: str) -> None:
        """Export metrics to JSON file."""
        metrics = self.get_metrics_summary()
        time_series = self.metrics_collector.get_time_series(minutes=60)  # Last hour
        
        export_data = {
            'summary': metrics,
            'time_series': time_series,
            'export_timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filename}")


def create_dashboard(router) -> RealtimeDashboard:
    """Create a real-time dashboard for the given router."""
    return RealtimeDashboard(router)