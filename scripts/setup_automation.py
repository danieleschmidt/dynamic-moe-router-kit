#!/usr/bin/env python3
"""
Setup automation for dynamic-moe-router-kit.

This script configures automated tasks, cron jobs, and monitoring
for the project maintenance and metrics collection.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomationSetup:
    """Configure project automation."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.project_root = Path.cwd()
        self.scripts_dir = self.project_root / "scripts"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load automation configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            sys.exit(1)
    
    def setup_cron_jobs(self) -> bool:
        """Setup cron jobs for automated tasks."""
        logger.info("Setting up cron jobs...")
        
        try:
            # Get current crontab
            result = subprocess.run(
                ['crontab', '-l'], 
                capture_output=True, 
                text=True
            )
            
            current_crontab = result.stdout if result.returncode == 0 else ""
            
            # Define cron jobs
            cron_jobs = [
                "# Dynamic MoE Router Automation",
                "# Metrics collection every 4 hours",
                f"0 */4 * * * cd {self.project_root} && python scripts/collect_metrics.py --output reports/metrics/daily/metrics-$(date +\\%Y\\%m\\%d_\\%H\\%M).json",
                "",
                "# Daily maintenance at 2 AM",
                f"0 2 * * * cd {self.project_root} && python scripts/maintenance_tasks.py --tasks security cleanup --output reports/maintenance/daily-$(date +\\%Y\\%m\\%d).json",
                "",
                "# Weekly comprehensive maintenance on Sundays at 3 AM",
                f"0 3 * * 0 cd {self.project_root} && python scripts/maintenance_tasks.py --tasks all --output reports/maintenance/weekly-$(date +\\%Y\\%m\\%d).json",
                "",
                "# Monthly dependency updates on 1st at 4 AM",
                f"0 4 1 * * cd {self.project_root} && python scripts/maintenance_tasks.py --tasks dependencies --update-type minor --output reports/maintenance/monthly-$(date +\\%Y\\%m\\%d).json",
                ""
            ]
            
            # Check if our jobs are already present
            has_automation = "Dynamic MoE Router Automation" in current_crontab
            
            if not has_automation:
                # Add our cron jobs
                new_crontab = current_crontab + "\n" + "\n".join(cron_jobs)
                
                # Write new crontab
                process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
                process.communicate(input=new_crontab)
                
                if process.returncode == 0:
                    logger.info("✅ Cron jobs added successfully")
                    return True
                else:
                    logger.error("❌ Failed to add cron jobs")
                    return False
            else:
                logger.info("ℹ️ Cron jobs already configured")
                return True
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup cron jobs: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting up cron jobs: {e}")
            return False
    
    def setup_git_hooks(self) -> bool:
        """Setup Git hooks for automation."""
        logger.info("Setting up Git hooks...")
        
        try:
            hooks_dir = self.project_root / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)
            
            # Pre-commit hook
            pre_commit_hook = hooks_dir / "pre-commit"
            pre_commit_content = f"""#!/bin/bash
# Dynamic MoE Router pre-commit hook

set -e

echo "🔍 Running pre-commit checks..."

# Run code quality checks
python scripts/maintenance_tasks.py --tasks quality

# Run quick security scan
python scripts/maintenance_tasks.py --tasks security

# Collect basic metrics
python scripts/collect_metrics.py --categories git test --output reports/pre-commit-metrics.json

echo "✅ Pre-commit checks passed"
"""
            
            with open(pre_commit_hook, 'w') as f:
                f.write(pre_commit_content)
            
            pre_commit_hook.chmod(0o755)
            
            # Post-merge hook
            post_merge_hook = hooks_dir / "post-merge"
            post_merge_content = f"""#!/bin/bash
# Dynamic MoE Router post-merge hook

set -e

echo "📊 Collecting post-merge metrics..."

# Collect comprehensive metrics after merge
python scripts/collect_metrics.py --output reports/post-merge-metrics.json

# Run maintenance tasks
python scripts/maintenance_tasks.py --tasks cleanup --output reports/post-merge-maintenance.json

echo "✅ Post-merge tasks completed"
"""
            
            with open(post_merge_hook, 'w') as f:
                f.write(post_merge_content)
            
            post_merge_hook.chmod(0o755)
            
            logger.info("✅ Git hooks configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Git hooks: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Setup required directories for automation."""
        logger.info("Setting up automation directories...")
        
        try:
            directories = [
                "reports",
                "reports/metrics",
                "reports/metrics/daily",
                "reports/metrics/weekly", 
                "reports/metrics/monthly",
                "reports/maintenance",
                "reports/maintenance/daily",
                "reports/maintenance/weekly",
                "reports/maintenance/monthly",
                "logs",
                "logs/automation",
                "cache",
                "cache/metrics",
                "cache/benchmarks"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Add .gitkeep to maintain directory structure
                gitkeep = dir_path / ".gitkeep"
                if not gitkeep.exists():
                    gitkeep.touch()
            
            # Create .gitignore for reports directory
            reports_gitignore = self.project_root / "reports" / ".gitignore"
            gitignore_content = """# Ignore generated reports but keep structure
*.json
*.html
*.pdf
*.csv
!.gitkeep

# Keep monthly reports for archival
!monthly/
monthly/*.json
!monthly/.gitkeep
"""
            
            with open(reports_gitignore, 'w') as f:
                f.write(gitignore_content)
            
            logger.info("✅ Directory structure created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup directories: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Setup monitoring and alerting."""
        logger.info("Setting up monitoring...")
        
        try:
            # Create monitoring configuration
            monitoring_config = {
                "alerts": {
                    "email": {
                        "enabled": False,
                        "smtp_server": "smtp.example.com",
                        "smtp_port": 587,
                        "username": "",
                        "recipients": []
                    },
                    "slack": {
                        "enabled": False,
                        "webhook_url": "",
                        "channel": "#alerts"
                    }
                },
                "thresholds": {
                    "critical_metrics": [
                        "security_vulnerabilities >= 3",
                        "test_coverage < 70",
                        "ci_success_rate < 85"
                    ],
                    "warning_metrics": [
                        "security_vulnerabilities >= 1",
                        "test_coverage < 80",
                        "routing_latency_p95 > 25"
                    ]
                },
                "notification_schedule": {
                    "critical": "immediate",
                    "warning": "daily_digest",
                    "info": "weekly_digest"
                }
            }
            
            config_path = self.project_root / "config" / "monitoring.json"
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(monitoring_config, f, indent=2)
            
            # Create alerting script
            alerting_script = self.scripts_dir / "alerting.py"
            alerting_content = """#!/usr/bin/env python3
'''
Alerting system for dynamic-moe-router-kit automation.
'''

import json
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from pathlib import Path

def send_alert(alert_type, message, config_path="config/monitoring.json"):
    '''Send alert based on configuration.'''
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Send email alert
        if config['alerts']['email']['enabled']:
            send_email_alert(message, config['alerts']['email'])
        
        # Send Slack alert
        if config['alerts']['slack']['enabled']:
            send_slack_alert(message, config['alerts']['slack'])
            
    except Exception as e:
        logging.error(f"Failed to send alert: {e}")

def send_email_alert(message, email_config):
    '''Send email alert.'''
    # Implementation for email alerts
    pass

def send_slack_alert(message, slack_config):
    '''Send Slack alert.'''
    try:
        payload = {
            "channel": slack_config['channel'],
            "text": message,
            "username": "Dynamic MoE Router Bot"
        }
        
        response = requests.post(
            slack_config['webhook_url'],
            json=payload
        )
        response.raise_for_status()
        
    except Exception as e:
        logging.error(f"Failed to send Slack alert: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        send_alert(sys.argv[1], sys.argv[2])
"""
            
            with open(alerting_script, 'w') as f:
                f.write(alerting_content)
            
            alerting_script.chmod(0o755)
            
            logger.info("✅ Monitoring configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            return False
    
    def create_systemd_services(self) -> bool:
        """Create systemd services for automation (Linux only)."""
        if os.name != 'posix':
            logger.info("ℹ️ Skipping systemd setup on non-Linux system")
            return True
        
        logger.info("Creating systemd services...")
        
        try:
            # Metrics collection service
            service_content = f"""[Unit]
Description=Dynamic MoE Router Metrics Collection
After=network.target

[Service]
Type=oneshot
User={os.getenv('USER', 'root')}
WorkingDirectory={self.project_root}
ExecStart=/usr/bin/python3 {self.scripts_dir}/collect_metrics.py --output reports/metrics/service-metrics.json
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
            
            # Timer for metrics collection
            timer_content = """[Unit]
Description=Run Dynamic MoE Router Metrics Collection every 4 hours
Requires=dynamic-moe-metrics.service

[Timer]
OnCalendar=*-*-* 00/4:00:00
Persistent=true

[Install]
WantedBy=timers.target
"""
            
            # Write service files (would need sudo access)
            logger.info("Service files ready - manual installation required:")
            logger.info("sudo cp config/systemd/dynamic-moe-metrics.service /etc/systemd/system/")
            logger.info("sudo cp config/systemd/dynamic-moe-metrics.timer /etc/systemd/system/")
            logger.info("sudo systemctl enable dynamic-moe-metrics.timer")
            logger.info("sudo systemctl start dynamic-moe-metrics.timer")
            
            # Create service files in config directory
            systemd_dir = self.project_root / "config" / "systemd"
            systemd_dir.mkdir(parents=True, exist_ok=True)
            
            with open(systemd_dir / "dynamic-moe-metrics.service", 'w') as f:
                f.write(service_content)
            
            with open(systemd_dir / "dynamic-moe-metrics.timer", 'w') as f:
                f.write(timer_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create systemd services: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Setup environment configuration."""
        logger.info("Setting up automation environment...")
        
        try:
            # Create automation environment file
            env_content = f"""# Dynamic MoE Router Automation Environment
# Generated on {datetime.now().isoformat()}

# Project paths
PROJECT_ROOT={self.project_root}
SCRIPTS_DIR={self.scripts_dir}
REPORTS_DIR={self.project_root}/reports
LOGS_DIR={self.project_root}/logs

# Python settings
PYTHONPATH={self.project_root}/src
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1

# Automation settings
AUTOMATION_ENABLED=true
METRICS_RETENTION_DAYS=90
REPORTS_RETENTION_DAYS=30
LOG_LEVEL=INFO

# GitHub settings (set these manually)
# GITHUB_TOKEN=your_github_token_here
# GITHUB_REPOSITORY={self.config['project']['repository']}

# Notification settings (set these manually)
# SLACK_WEBHOOK_URL=your_slack_webhook_url
# EMAIL_SMTP_SERVER=smtp.example.com
# EMAIL_USERNAME=your_email@example.com
# EMAIL_PASSWORD=your_email_password

# External services (set these manually)
# CODECOV_TOKEN=your_codecov_token
# SONAR_TOKEN=your_sonar_token
"""
            
            env_file = self.project_root / ".env.automation"
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            # Create setup script
            setup_script = self.scripts_dir / "setup_env.sh"
            setup_content = f"""#!/bin/bash
# Environment setup for Dynamic MoE Router automation

set -e

echo "🔧 Setting up automation environment..."

# Load automation environment
if [ -f .env.automation ]; then
    export $(cat .env.automation | xargs)
fi

# Create required directories
mkdir -p ${{REPORTS_DIR}} ${{LOGS_DIR}}

# Install Python dependencies
pip install -r requirements.txt

# Setup pre-commit hooks
if command -v pre-commit &> /dev/null; then
    pre-commit install
fi

# Test metrics collection
echo "🧪 Testing metrics collection..."
python scripts/collect_metrics.py --categories git --output reports/test-metrics.json

echo "✅ Automation environment setup complete"

# Show next steps
echo ""
echo "📋 Next steps:"
echo "1. Set GitHub token: export GITHUB_TOKEN=your_token"
echo "2. Configure Slack webhook: export SLACK_WEBHOOK_URL=your_webhook"
echo "3. Run: python scripts/setup_automation.py --enable-cron"
echo "4. Monitor: tail -f logs/automation.log"
"""
            
            with open(setup_script, 'w') as f:
                f.write(setup_content)
            
            setup_script.chmod(0o755)
            
            logger.info("✅ Environment configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate automation setup."""
        logger.info("Validating automation setup...")
        
        try:
            checks = []
            
            # Check required files
            required_files = [
                "scripts/collect_metrics.py",
                "scripts/maintenance_tasks.py",
                ".github/project-metrics.json",
                ".env.automation"
            ]
            
            for file_path in required_files:
                path = self.project_root / file_path
                checks.append(("File exists", file_path, path.exists()))
            
            # Check directories
            required_dirs = [
                "reports",
                "reports/metrics",
                "reports/maintenance",
                "logs"
            ]
            
            for dir_path in required_dirs:
                path = self.project_root / dir_path
                checks.append(("Directory exists", dir_path, path.exists()))
            
            # Check executable permissions
            executable_files = [
                "scripts/collect_metrics.py",
                "scripts/maintenance_tasks.py"
            ]
            
            for file_path in executable_files:
                path = self.project_root / file_path
                is_executable = path.exists() and os.access(path, os.X_OK)
                checks.append(("Executable", file_path, is_executable))
            
            # Print validation results
            all_passed = True
            for check_type, item, passed in checks:
                status = "✅" if passed else "❌"
                logger.info(f"{status} {check_type}: {item}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                logger.info("🎉 All validation checks passed")
            else:
                logger.error("❌ Some validation checks failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Setup project automation')
    parser.add_argument(
        '--enable-cron',
        action='store_true',
        help='Enable cron jobs'
    )
    parser.add_argument(
        '--enable-git-hooks',
        action='store_true',
        help='Enable git hooks'
    )
    parser.add_argument(
        '--enable-systemd',
        action='store_true',
        help='Create systemd services'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Setup all automation components'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing setup'
    )
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = AutomationSetup()
    
    if args.validate_only:
        success = setup.validate_setup()
        sys.exit(0 if success else 1)
    
    # Setup components
    tasks = []
    
    if args.all or not any([args.enable_cron, args.enable_git_hooks, args.enable_systemd]):
        tasks = ['directories', 'environment', 'monitoring', 'git_hooks', 'systemd']
        if args.all or args.enable_cron:
            tasks.append('cron')
    else:
        tasks.append('directories')
        tasks.append('environment')
        tasks.append('monitoring')
        
        if args.enable_git_hooks:
            tasks.append('git_hooks')
        if args.enable_systemd:
            tasks.append('systemd')
        if args.enable_cron:
            tasks.append('cron')
    
    # Run setup tasks
    success_count = 0
    total_tasks = len(tasks)
    
    for task in tasks:
        if task == 'directories':
            success = setup.setup_directories()
        elif task == 'environment':
            success = setup.setup_environment()
        elif task == 'monitoring':
            success = setup.setup_monitoring()
        elif task == 'git_hooks':
            success = setup.setup_git_hooks()
        elif task == 'systemd':
            success = setup.create_systemd_services()
        elif task == 'cron':
            success = setup.setup_cron_jobs()
        else:
            success = False
        
        if success:
            success_count += 1
    
    # Validate setup
    validation_success = setup.validate_setup()
    
    # Summary
    print(f"\n🔧 Automation Setup Summary:")
    print(f"Setup Tasks: {success_count}/{total_tasks} successful")
    print(f"Validation: {'✅ Passed' if validation_success else '❌ Failed'}")
    
    if success_count == total_tasks and validation_success:
        print("\n🎉 Automation setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Set required environment variables in .env.automation")
        print("2. Test metrics collection: python scripts/collect_metrics.py")
        print("3. Test maintenance: python scripts/maintenance_tasks.py --tasks cleanup")
        print("4. Monitor logs: tail -f logs/automation.log")
    else:
        print("\n❌ Setup completed with errors - check logs above")
        sys.exit(1)


if __name__ == '__main__':
    main()