#!/usr/bin/env python3
"""
Comprehensive metrics collection script for dynamic-moe-router-kit.

This script collects various project metrics from different sources and
generates reports for monitoring project health and performance.
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import subprocess
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a collected metric value."""
    name: str
    value: float
    timestamp: str
    category: str
    status: str  # 'ok', 'warning', 'critical'
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


class MetricsCollector:
    """Main metrics collection orchestrator."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize metrics collector with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.metrics: List[MetricValue] = []
        
        # Initialize API clients
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.codecov_token = os.getenv('CODECOV_TOKEN')
        self.sonar_token = os.getenv('SONAR_TOKEN')
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            sys.exit(1)
    
    def _make_github_request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make authenticated GitHub API request."""
        if not self.github_token:
            logger.warning("GitHub token not available, skipping GitHub metrics")
            return None
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(f"https://api.github.com{endpoint}", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            return None
    
    def _get_repository_info(self) -> Dict[str, Any]:
        """Get basic repository information."""
        repo_url = self.config['project']['repository']
        repo_path = repo_url.replace('https://github.com/', '')
        
        return self._make_github_request(f"/repos/{repo_path}") or {}
    
    def collect_github_metrics(self) -> None:
        """Collect metrics from GitHub API."""
        logger.info("Collecting GitHub metrics...")
        
        repo_info = self._get_repository_info()
        if not repo_info:
            return
        
        # Basic repository metrics
        self._add_metric(
            "github_stars", 
            repo_info.get('stargazers_count', 0),
            "community"
        )
        
        self._add_metric(
            "github_forks",
            repo_info.get('forks_count', 0),
            "community"
        )
        
        self._add_metric(
            "github_watchers",
            repo_info.get('watchers_count', 0),
            "community"
        )
        
        # Issues and PRs
        repo_path = repo_info.get('full_name', '')
        if repo_path:
            issues_data = self._make_github_request(f"/repos/{repo_path}/issues?state=open")
            if issues_data:
                open_issues = len([i for i in issues_data if 'pull_request' not in i])
                open_prs = len([i for i in issues_data if 'pull_request' in i])
                
                self._add_metric("open_issues", open_issues, "development")
                self._add_metric("open_pull_requests", open_prs, "development")
        
        # Contributors
        contributors_data = self._make_github_request(f"/repos/{repo_path}/contributors")
        if contributors_data:
            self._add_metric(
                "contributor_count",
                len(contributors_data),
                "community"
            )
    
    def collect_git_metrics(self) -> None:
        """Collect metrics from git repository."""
        logger.info("Collecting Git metrics...")
        
        try:
            # Commit frequency (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            result = subprocess.run([
                'git', 'rev-list', '--count', f'--since={thirty_days_ago}', 'HEAD'
            ], capture_output=True, text=True, check=True)
            
            commits_30_days = int(result.stdout.strip())
            commits_per_week = (commits_30_days / 30) * 7
            
            self._add_metric("commit_frequency", commits_per_week, "development")
            
            # Repository size
            result = subprocess.run([
                'git', 'count-objects', '-v'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            size_kb = 0
            for line in lines:
                if line.startswith('size '):
                    size_kb = int(line.split()[1])
                    break
            
            self._add_metric("repository_size_mb", size_kb / 1024, "development")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
        except Exception as e:
            logger.error(f"Error collecting git metrics: {e}")
    
    def collect_test_metrics(self) -> None:
        """Collect test coverage and quality metrics."""
        logger.info("Collecting test metrics...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', 
                '--cov=dynamic_moe_router',
                '--cov-report=json',
                '--cov-report=term-missing',
                'tests/',
                '-q'
            ], capture_output=True, text=True)
            
            # Parse coverage report
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                self._add_metric("test_coverage", total_coverage, "quality")
                
                # Line counts
                total_lines = coverage_data['totals']['num_statements']
                self._add_metric("total_lines", total_lines, "development")
        
        except Exception as e:
            logger.error(f"Error collecting test metrics: {e}")
    
    def collect_security_metrics(self) -> None:
        """Collect security-related metrics."""
        logger.info("Collecting security metrics...")
        
        try:
            # Run safety check
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self._add_metric("security_vulnerabilities", 0, "quality")
            else:
                try:
                    safety_data = json.loads(result.stdout)
                    vuln_count = len(safety_data)
                    self._add_metric("security_vulnerabilities", vuln_count, "quality")
                except json.JSONDecodeError:
                    logger.warning("Could not parse safety check output")
            
            # Run bandit security scan
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True)
            
            try:
                bandit_data = json.loads(result.stdout)
                high_severity = len([r for r in bandit_data.get('results', []) 
                                   if r.get('issue_severity') == 'HIGH'])
                medium_severity = len([r for r in bandit_data.get('results', [])
                                     if r.get('issue_severity') == 'MEDIUM'])
                
                self._add_metric("security_issues_high", high_severity, "quality")
                self._add_metric("security_issues_medium", medium_severity, "quality")
                
            except json.JSONDecodeError:
                logger.warning("Could not parse bandit output")
        
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
    
    def collect_performance_metrics(self) -> None:
        """Collect performance benchmarks."""
        logger.info("Collecting performance metrics...")
        
        try:
            # Run performance benchmarks
            result = subprocess.run([
                'python', '-m', 'pytest',
                'tests/performance/',
                '--benchmark-json=benchmark-results.json',
                '-q'
            ], capture_output=True, text=True)
            
            # Parse benchmark results
            benchmark_file = Path('benchmark-results.json')
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                
                for bench in benchmark_data.get('benchmarks', []):
                    name = bench['name']
                    mean_time = bench['stats']['mean']
                    
                    # Convert to appropriate metrics
                    if 'routing' in name.lower():
                        # Convert to milliseconds
                        latency_ms = mean_time * 1000
                        self._add_metric("routing_latency_mean", latency_ms, "performance")
                    
                    if 'throughput' in name.lower():
                        # Assume result is in tokens/second
                        self._add_metric("throughput", 1/mean_time, "performance")
        
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    def collect_dependency_metrics(self) -> None:
        """Collect dependency freshness metrics."""
        logger.info("Collecting dependency metrics...")
        
        try:
            # Check for outdated packages
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            outdated_packages = json.loads(result.stdout)
            
            # Get total package count
            result_all = subprocess.run([
                'pip', 'list', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            all_packages = json.loads(result_all.stdout)
            
            if all_packages:
                freshness = (1 - len(outdated_packages) / len(all_packages)) * 100
                self._add_metric("dependency_freshness", freshness, "quality")
            
            self._add_metric("outdated_dependencies", len(outdated_packages), "quality")
        
        except Exception as e:
            logger.error(f"Error collecting dependency metrics: {e}")
    
    def collect_pypi_metrics(self) -> None:
        """Collect PyPI download statistics."""
        logger.info("Collecting PyPI metrics...")
        
        package_name = self.config['project']['name']
        
        try:
            # Get download stats from pypistats.org
            response = requests.get(
                f"https://pypistats.org/api/packages/{package_name}/recent"
            )
            
            if response.status_code == 200:
                data = response.json()
                last_month = data['data'].get('last_month', 0)
                last_week = data['data'].get('last_week', 0)
                last_day = data['data'].get('last_day', 0)
                
                self._add_metric("pypi_downloads_month", last_month, "community")
                self._add_metric("pypi_downloads_week", last_week, "community")
                self._add_metric("pypi_downloads_day", last_day, "community")
        
        except Exception as e:
            logger.error(f"Error collecting PyPI metrics: {e}")
    
    def _add_metric(self, name: str, value: float, category: str) -> None:
        """Add a metric value with status evaluation."""
        # Get thresholds from config
        metric_config = self.config.get('categories', {}).get(category, {}).get('metrics', {}).get(name, {})
        warning_threshold = metric_config.get('threshold_warning')
        critical_threshold = metric_config.get('threshold_critical')
        
        # Determine status
        status = 'ok'
        if critical_threshold is not None and value <= critical_threshold:
            status = 'critical'
        elif warning_threshold is not None and value <= warning_threshold:
            status = 'warning'
        
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.utcnow().isoformat(),
            category=category,
            status=status,
            threshold_warning=warning_threshold,
            threshold_critical=critical_threshold
        )
        
        self.metrics.append(metric)
        logger.info(f"Collected {name}: {value} ({status})")
    
    def generate_report(self, output_format: str = 'json') -> str:
        """Generate metrics report in specified format."""
        timestamp = datetime.utcnow().isoformat()
        
        report_data = {
            'generated_at': timestamp,
            'project': self.config['project'],
            'summary': {
                'total_metrics': len(self.metrics),
                'critical_count': len([m for m in self.metrics if m.status == 'critical']),
                'warning_count': len([m for m in self.metrics if m.status == 'warning']),
                'ok_count': len([m for m in self.metrics if m.status == 'ok'])
            },
            'metrics': [asdict(metric) for metric in self.metrics]
        }
        
        if output_format == 'json':
            return json.dumps(report_data, indent=2)
        elif output_format == 'markdown':
            return self._generate_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """Generate markdown format report."""
        md = []
        md.append(f"# Metrics Report")
        md.append(f"Generated: {data['generated_at']}")
        md.append("")
        
        md.append("## Summary")
        summary = data['summary']
        md.append(f"- Total Metrics: {summary['total_metrics']}")
        md.append(f"- 🔴 Critical: {summary['critical_count']}")
        md.append(f"- 🟡 Warning: {summary['warning_count']}")
        md.append(f"- 🟢 OK: {summary['ok_count']}")
        md.append("")
        
        # Group metrics by category
        categories = {}
        for metric in data['metrics']:
            cat = metric['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(metric)
        
        for category, metrics in categories.items():
            md.append(f"## {category.title()} Metrics")
            md.append("| Metric | Value | Status |")
            md.append("|--------|-------|--------|")
            
            for metric in metrics:
                status_emoji = {
                    'ok': '🟢',
                    'warning': '🟡', 
                    'critical': '🔴'
                }.get(metric['status'], '⚪')
                
                md.append(f"| {metric['name']} | {metric['value']:.2f} | {status_emoji} {metric['status']} |")
            md.append("")
        
        return '\n'.join(md)
    
    def save_report(self, output_file: str, output_format: str = 'json') -> None:
        """Save metrics report to file."""
        report = self.generate_report(output_format)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_file}")
    
    def run_collection(self, categories: Optional[List[str]] = None) -> None:
        """Run metrics collection for specified categories."""
        all_categories = [
            'github', 'git', 'test', 'security', 
            'performance', 'dependency', 'pypi'
        ]
        
        if categories is None:
            categories = all_categories
        
        # Collection methods mapping
        collectors = {
            'github': self.collect_github_metrics,
            'git': self.collect_git_metrics,
            'test': self.collect_test_metrics,
            'security': self.collect_security_metrics,
            'performance': self.collect_performance_metrics,
            'dependency': self.collect_dependency_metrics,
            'pypi': self.collect_pypi_metrics
        }
        
        for category in categories:
            if category in collectors:
                try:
                    collectors[category]()
                except Exception as e:
                    logger.error(f"Error collecting {category} metrics: {e}")
            else:
                logger.warning(f"Unknown category: {category}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument(
        '--categories', 
        nargs='+', 
        help='Categories to collect (default: all)'
    )
    parser.add_argument(
        '--output', 
        default='reports/metrics-{timestamp}.json',
        help='Output file path (default: reports/metrics-{timestamp}.json)'
    )
    parser.add_argument(
        '--format', 
        choices=['json', 'markdown'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--config',
        default='.github/project-metrics.json',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # Replace timestamp placeholder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output.replace('{timestamp}', timestamp)
    
    # Initialize collector
    collector = MetricsCollector(args.config)
    
    # Run collection
    logger.info("Starting metrics collection...")
    start_time = time.time()
    
    collector.run_collection(args.categories)
    
    # Generate and save report
    collector.save_report(output_file, args.format)
    
    duration = time.time() - start_time
    logger.info(f"Metrics collection completed in {duration:.2f} seconds")
    
    # Print summary
    summary = {
        'total': len(collector.metrics),
        'critical': len([m for m in collector.metrics if m.status == 'critical']),
        'warning': len([m for m in collector.metrics if m.status == 'warning']),
        'ok': len([m for m in collector.metrics if m.status == 'ok'])
    }
    
    print(f"\n📊 Metrics Collection Summary:")
    print(f"Total: {summary['total']}")
    print(f"🔴 Critical: {summary['critical']}")
    print(f"🟡 Warning: {summary['warning']}")
    print(f"🟢 OK: {summary['ok']}")
    print(f"\nReport saved: {output_file}")
    
    # Exit with error code if critical issues found
    if summary['critical'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()