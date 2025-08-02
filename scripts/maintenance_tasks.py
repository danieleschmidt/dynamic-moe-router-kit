#!/usr/bin/env python3
"""
Automated maintenance tasks for dynamic-moe-router-kit.

This script performs various maintenance tasks including dependency updates,
code quality improvements, and repository cleanup.
"""

import os
import sys
import json
import subprocess
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaintenanceTask:
    """Base class for maintenance tasks."""
    
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.error_message = None
    
    def run(self) -> bool:
        """Execute the maintenance task."""
        try:
            logger.info(f"Running task: {self.name}")
            result = self._execute()
            self.success = result
            if result:
                logger.info(f"✅ Task completed: {self.name}")
            else:
                logger.warning(f"⚠️ Task failed: {self.name}")
            return result
        except Exception as e:
            self.success = False
            self.error_message = str(e)
            logger.error(f"❌ Task error: {self.name} - {e}")
            return False
    
    def _execute(self) -> bool:
        """Override this method to implement the task."""
        raise NotImplementedError


class DependencyUpdateTask(MaintenanceTask):
    """Update project dependencies."""
    
    def __init__(self, update_type: str = 'minor'):
        super().__init__(f"dependency_update_{update_type}")
        self.update_type = update_type
    
    def _execute(self) -> bool:
        """Update dependencies based on type."""
        try:
            # Check for outdated packages
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            outdated = json.loads(result.stdout)
            
            if not outdated:
                logger.info("All dependencies are up to date")
                return True
            
            logger.info(f"Found {len(outdated)} outdated packages")
            
            # Update packages based on type
            packages_to_update = []
            
            for pkg in outdated:
                current_version = pkg['version']
                latest_version = pkg['latest_version']
                
                if self.update_type == 'patch':
                    # Only patch updates
                    if self._is_patch_update(current_version, latest_version):
                        packages_to_update.append(pkg['name'])
                elif self.update_type == 'minor':
                    # Minor and patch updates
                    if self._is_minor_or_patch_update(current_version, latest_version):
                        packages_to_update.append(pkg['name'])
                elif self.update_type == 'all':
                    packages_to_update.append(pkg['name'])
            
            if packages_to_update:
                logger.info(f"Updating {len(packages_to_update)} packages: {packages_to_update}")
                
                # Update packages
                subprocess.run([
                    'pip', 'install', '--upgrade'
                ] + packages_to_update, check=True)
                
                # Run tests to verify compatibility
                test_result = subprocess.run([
                    'python', '-m', 'pytest', 'tests/', '-x', '--tb=short'
                ], capture_output=True, text=True)
                
                if test_result.returncode != 0:
                    logger.warning("Tests failed after dependency update")
                    return False
                
                return True
            else:
                logger.info("No packages to update for this update type")
                return True
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Dependency update failed: {e}")
            return False
    
    def _is_patch_update(self, current: str, latest: str) -> bool:
        """Check if update is a patch version."""
        try:
            curr_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            return (curr_parts[0] == latest_parts[0] and 
                   curr_parts[1] == latest_parts[1] and
                   curr_parts[2] < latest_parts[2])
        except:
            return False
    
    def _is_minor_or_patch_update(self, current: str, latest: str) -> bool:
        """Check if update is a minor or patch version."""
        try:
            curr_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            return (curr_parts[0] == latest_parts[0] and
                   curr_parts[1] <= latest_parts[1])
        except:
            return False


class SecurityScanTask(MaintenanceTask):
    """Run security scans and update vulnerable dependencies."""
    
    def __init__(self):
        super().__init__("security_scan")
    
    def _execute(self) -> bool:
        """Run security scans."""
        try:
            vulnerabilities_found = False
            
            # Run safety check
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                try:
                    safety_data = json.loads(result.stdout)
                    if safety_data:
                        logger.warning(f"Safety found {len(safety_data)} vulnerabilities")
                        vulnerabilities_found = True
                        
                        # Try to update vulnerable packages
                        for vuln in safety_data:
                            package = vuln.get('package_name')
                            if package:
                                logger.info(f"Updating vulnerable package: {package}")
                                subprocess.run([
                                    'pip', 'install', '--upgrade', package
                                ], check=True)
                except json.JSONDecodeError:
                    logger.warning("Could not parse safety output")
            
            # Run bandit security scan
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True)
            
            try:
                bandit_data = json.loads(result.stdout)
                issues = bandit_data.get('results', [])
                high_issues = [i for i in issues if i.get('issue_severity') == 'HIGH']
                
                if high_issues:
                    logger.warning(f"Bandit found {len(high_issues)} high severity issues")
                    vulnerabilities_found = True
            except json.JSONDecodeError:
                logger.warning("Could not parse bandit output")
            
            return not vulnerabilities_found
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return False


class CodeQualityTask(MaintenanceTask):
    """Improve code quality through automated fixes."""
    
    def __init__(self):
        super().__init__("code_quality")
    
    def _execute(self) -> bool:
        """Run code quality improvements."""
        try:
            changes_made = False
            
            # Run black formatting
            result = subprocess.run([
                'black', '--check', 'src/', 'tests/'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.info("Running black formatter")
                subprocess.run(['black', 'src/', 'tests/'], check=True)
                changes_made = True
            
            # Run isort
            result = subprocess.run([
                'isort', '--check-only', 'src/', 'tests/'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.info("Running isort")
                subprocess.run(['isort', 'src/', 'tests/'], check=True)
                changes_made = True
            
            # Run ruff with fixes
            result = subprocess.run([
                'ruff', 'check', 'src/', 'tests/', '--fix'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.info("Applied ruff fixes")
                changes_made = True
            
            if changes_made:
                logger.info("Code quality improvements applied")
            else:
                logger.info("No code quality issues found")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Code quality task failed: {e}")
            return False


class DocumentationUpdateTask(MaintenanceTask):
    """Update documentation and check for issues."""
    
    def __init__(self):
        super().__init__("documentation_update")
    
    def _execute(self) -> bool:
        """Update documentation."""
        try:
            # Check for broken links in documentation
            doc_files = list(Path('docs').rglob('*.md'))
            
            broken_links = []
            for doc_file in doc_files:
                links = self._extract_links(doc_file)
                for link in links:
                    if link.startswith('http') and not self._check_link(link):
                        broken_links.append((doc_file, link))
            
            if broken_links:
                logger.warning(f"Found {len(broken_links)} broken links")
                for file, link in broken_links:
                    logger.warning(f"Broken link in {file}: {link}")
                return False
            
            # Build documentation
            try:
                subprocess.run([
                    'sphinx-build', '-b', 'html', 'docs/', 'docs/_build/html'
                ], check=True, capture_output=True)
                logger.info("Documentation built successfully")
            except subprocess.CalledProcessError:
                logger.warning("Documentation build failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Documentation update failed: {e}")
            return False
    
    def _extract_links(self, file_path: Path) -> List[str]:
        """Extract HTTP links from markdown file."""
        import re
        
        try:
            content = file_path.read_text()
            # Simple regex to find HTTP links
            links = re.findall(r'http[s]?://[^\s\)]+', content)
            return links
        except Exception:
            return []
    
    def _check_link(self, url: str) -> bool:
        """Check if URL is accessible."""
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False


class RepositoryCleanupTask(MaintenanceTask):
    """Clean up repository artifacts and temporary files."""
    
    def __init__(self):
        super().__init__("repository_cleanup")
    
    def _execute(self) -> bool:
        """Clean up repository."""
        try:
            cleaned_items = []
            
            # Remove Python cache files
            cache_dirs = list(Path('.').rglob('__pycache__'))
            for cache_dir in cache_dirs:
                if cache_dir.is_dir():
                    import shutil
                    shutil.rmtree(cache_dir)
                    cleaned_items.append(str(cache_dir))
            
            # Remove .pyc files
            pyc_files = list(Path('.').rglob('*.pyc'))
            for pyc_file in pyc_files:
                pyc_file.unlink()
                cleaned_items.append(str(pyc_file))
            
            # Remove pytest cache
            pytest_cache = Path('.pytest_cache')
            if pytest_cache.exists():
                import shutil
                shutil.rmtree(pytest_cache)
                cleaned_items.append(str(pytest_cache))
            
            # Remove coverage files
            coverage_files = ['.coverage', 'coverage.xml', 'htmlcov']
            for cov_file in coverage_files:
                path = Path(cov_file)
                if path.exists():
                    if path.is_dir():
                        import shutil
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    cleaned_items.append(str(path))
            
            # Remove build artifacts
            build_dirs = ['build', 'dist', '*.egg-info']
            for build_pattern in build_dirs:
                for path in Path('.').glob(build_pattern):
                    if path.exists():
                        if path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        cleaned_items.append(str(path))
            
            if cleaned_items:
                logger.info(f"Cleaned {len(cleaned_items)} items")
            else:
                logger.info("Repository already clean")
            
            return True
            
        except Exception as e:
            logger.error(f"Repository cleanup failed: {e}")
            return False


class PerformanceMonitoringTask(MaintenanceTask):
    """Monitor and validate performance metrics."""
    
    def __init__(self):
        super().__init__("performance_monitoring")
    
    def _execute(self) -> bool:
        """Monitor performance."""
        try:
            # Run performance benchmarks
            result = subprocess.run([
                'python', '-m', 'pytest',
                'tests/performance/',
                '--benchmark-json=benchmark-results.json',
                '-q'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning("Performance benchmarks failed")
                return False
            
            # Analyze benchmark results
            benchmark_file = Path('benchmark-results.json')
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    data = json.load(f)
                
                # Check for performance regressions
                regressions = []
                for bench in data.get('benchmarks', []):
                    name = bench['name']
                    mean_time = bench['stats']['mean']
                    
                    # Define performance thresholds
                    thresholds = {
                        'routing': 0.050,  # 50ms
                        'complexity': 0.010,  # 10ms
                        'expert_selection': 0.020,  # 20ms
                    }
                    
                    for keyword, threshold in thresholds.items():
                        if keyword in name.lower() and mean_time > threshold:
                            regressions.append((name, mean_time, threshold))
                
                if regressions:
                    logger.warning(f"Found {len(regressions)} performance regressions")
                    for name, actual, expected in regressions:
                        logger.warning(f"{name}: {actual:.3f}s > {expected:.3f}s")
                    return False
                
                logger.info("All performance benchmarks passed")
                return True
            else:
                logger.warning("Benchmark results not found")
                return False
                
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            return False


class MaintenanceRunner:
    """Orchestrates maintenance tasks."""
    
    def __init__(self):
        self.tasks: List[MaintenanceTask] = []
        self.results: Dict[str, bool] = {}
    
    def add_task(self, task: MaintenanceTask):
        """Add a maintenance task."""
        self.tasks.append(task)
    
    def run_all(self) -> bool:
        """Run all maintenance tasks."""
        logger.info(f"Starting maintenance run with {len(self.tasks)} tasks")
        
        success_count = 0
        
        for task in self.tasks:
            success = task.run()
            self.results[task.name] = success
            if success:
                success_count += 1
        
        total_tasks = len(self.tasks)
        logger.info(f"Maintenance completed: {success_count}/{total_tasks} tasks successful")
        
        return success_count == total_tasks
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate maintenance report."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_tasks': len(self.tasks),
            'successful_tasks': sum(1 for result in self.results.values() if result),
            'failed_tasks': sum(1 for result in self.results.values() if not result),
            'task_results': {
                task.name: {
                    'success': task.success,
                    'error_message': task.error_message
                }
                for task in self.tasks
            }
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run maintenance tasks')
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=[
            'dependencies', 'security', 'quality', 
            'documentation', 'cleanup', 'performance', 'all'
        ],
        default=['all'],
        help='Tasks to run (default: all)'
    )
    parser.add_argument(
        '--update-type',
        choices=['patch', 'minor', 'all'],
        default='minor',
        help='Dependency update type (default: minor)'
    )
    parser.add_argument(
        '--output',
        default='reports/maintenance-{timestamp}.json',
        help='Output report file'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = MaintenanceRunner()
    
    # Add selected tasks
    task_types = args.tasks
    if 'all' in task_types:
        task_types = [
            'dependencies', 'security', 'quality',
            'documentation', 'cleanup', 'performance'
        ]
    
    for task_type in task_types:
        if task_type == 'dependencies':
            runner.add_task(DependencyUpdateTask(args.update_type))
        elif task_type == 'security':
            runner.add_task(SecurityScanTask())
        elif task_type == 'quality':
            runner.add_task(CodeQualityTask())
        elif task_type == 'documentation':
            runner.add_task(DocumentationUpdateTask())
        elif task_type == 'cleanup':
            runner.add_task(RepositoryCleanupTask())
        elif task_type == 'performance':
            runner.add_task(PerformanceMonitoringTask())
    
    # Run maintenance
    success = runner.run_all()
    
    # Generate report
    report = runner.generate_report()
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output.replace('{timestamp}', timestamp)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Maintenance report saved to {output_file}")
    
    # Print summary
    print(f"\n🔧 Maintenance Summary:")
    print(f"Total Tasks: {report['total_tasks']}")
    print(f"✅ Successful: {report['successful_tasks']}")
    print(f"❌ Failed: {report['failed_tasks']}")
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()