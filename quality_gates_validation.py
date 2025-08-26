"""Quality Gates Validation for Meta-Autonomous SDLC Evolution System.

This comprehensive validation ensures all quality gates pass for production deployment:
- Code quality and structure validation
- Security analysis and vulnerability scanning  
- Performance benchmarks and efficiency metrics
- Documentation completeness verification
- Research reproducibility validation

PRODUCTION READINESS: Validates enterprise-grade quality standards.
"""

import os
import sys
import time
import json
import ast
import inspect
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add source path
sys.path.append('src')


class QualityGateStatus(Enum):
    """Status of quality gate validation."""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"  
    WARNING = "⚠️ WARNING"
    SKIPPED = "⏭️ SKIPPED"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: QualityGateStatus
    score: float
    details: str
    recommendations: List[str]
    execution_time: float


class QualityGatesValidator:
    """Comprehensive quality gates validation system."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src" / "dynamic_moe_router"
        self.results: List[QualityGateResult] = []
        
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🛡️ QUALITY GATES VALIDATION")
        print("=" * 60)
        
        start_time = time.time()
        
        # Code Quality Gates
        self.validate_code_structure()
        self.validate_code_quality()
        self.validate_import_structure()
        
        # Security Gates  
        self.validate_security_practices()
        self.validate_dependency_security()
        
        # Performance Gates
        self.validate_performance_characteristics()
        self.validate_memory_efficiency()
        
        # Documentation Gates
        self.validate_documentation_completeness()
        self.validate_api_documentation()
        
        # Research Gates
        self.validate_research_reproducibility()
        self.validate_experimental_framework()
        
        # Deployment Gates
        self.validate_production_readiness()
        self.validate_configuration_management()
        
        total_time = time.time() - start_time
        
        return self._generate_quality_report(total_time)
    
    def validate_code_structure(self) -> QualityGateResult:
        """Validate code structure and organization."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check source directory exists
            if self.src_path.exists():
                score += 20
                details.append("Source directory structure exists")
            else:
                recommendations.append("Create proper source directory structure")
            
            # Check for core modules
            core_modules = [
                "meta_autonomous_evolution_engine.py",
                "breakthrough_research_framework.py", 
                "__init__.py"
            ]
            
            existing_modules = 0
            for module in core_modules:
                module_path = self.src_path / module
                if module_path.exists():
                    existing_modules += 1
                    score += 20
                    
                    # Check file size (should be substantial)
                    file_size = module_path.stat().st_size
                    if file_size > 1000:  # At least 1KB
                        score += 5
                        details.append(f"{module}: {file_size} bytes")
                    else:
                        recommendations.append(f"{module} seems too small")
            
            details.append(f"Core modules found: {existing_modules}/{len(core_modules)}")
            
            # Check for test files
            test_files = list(self.project_root.glob("test*.py"))
            if test_files:
                score += 15
                details.append(f"Test files found: {len(test_files)}")
            else:
                recommendations.append("Add comprehensive test files")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Code Structure",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Code Structure",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Validation failed: {e}",
                recommendations=["Fix code structure validation errors"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_code_quality(self) -> QualityGateResult:
        """Validate code quality metrics."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Analyze Python files
            python_files = list(self.src_path.glob("*.py"))
            if not python_files:
                return QualityGateResult(
                    gate_name="Code Quality",
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    details="No Python files found",
                    recommendations=["Add Python source files"],
                    execution_time=time.time() - start_time
                )
            
            total_lines = 0
            total_functions = 0
            total_classes = 0
            files_with_docstrings = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        total_lines += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                    
                    # Parse AST for analysis
                    tree = ast.parse(content)
                    
                    # Count functions and classes
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                    
                    # Check for module docstring
                    if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                        isinstance(tree.body[0].value, ast.Str)):
                        files_with_docstrings += 1
                    
                except Exception:
                    # Skip files that can't be parsed
                    continue
            
            # Calculate quality metrics
            if total_lines > 0:
                score += 20
                details.append(f"Total lines: {total_lines}")
            
            if total_functions > 0:
                score += 15
                details.append(f"Functions: {total_functions}")
            
            if total_classes > 0:
                score += 15  
                details.append(f"Classes: {total_classes}")
            
            # Documentation coverage
            doc_coverage = files_with_docstrings / len(python_files) * 100 if python_files else 0
            if doc_coverage >= 80:
                score += 25
            elif doc_coverage >= 50:
                score += 15
            else:
                recommendations.append("Improve docstring coverage")
            
            details.append(f"Documentation coverage: {doc_coverage:.1f}%")
            
            # Complexity analysis
            avg_functions_per_file = total_functions / len(python_files) if python_files else 0
            if 5 <= avg_functions_per_file <= 20:  # Good balance
                score += 15
                details.append(f"Avg functions per file: {avg_functions_per_file:.1f}")
            else:
                recommendations.append("Review function distribution across files")
            
            # File size analysis
            large_files = [f for f in python_files if f.stat().st_size > 50000]  # 50KB+
            if len(large_files) <= len(python_files) * 0.2:  # Max 20% large files
                score += 10
            else:
                recommendations.append("Consider splitting large files")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Code Quality", 
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Code Quality",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Analysis failed: {e}",
                recommendations=["Fix code quality analysis errors"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_import_structure(self) -> QualityGateResult:
        """Validate import structure and dependencies."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            init_file = self.src_path / "__init__.py"
            if init_file.exists():
                score += 30
                
                with open(init_file, 'r') as f:
                    init_content = f.read()
                
                # Check for proper exports
                if "__all__" in init_content:
                    score += 20
                    details.append("__all__ defined for explicit exports")
                else:
                    recommendations.append("Define __all__ for explicit API")
                
                # Count imports
                import_count = init_content.count("from .")
                if import_count > 5:
                    score += 20
                    details.append(f"Relative imports: {import_count}")
                else:
                    recommendations.append("Add more comprehensive imports")
                
                # Check for version info
                if "__version__" in init_content:
                    score += 15
                    details.append("Version information included")
                else:
                    recommendations.append("Add version information")
                
                # Check for dependency handling
                if "ImportError" in init_content or "try:" in init_content:
                    score += 15
                    details.append("Graceful dependency handling")
                else:
                    recommendations.append("Add graceful dependency handling")
            else:
                recommendations.append("Create __init__.py for proper package structure")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Import Structure",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Import Structure",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Analysis failed: {e}",
                recommendations=["Fix import structure analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_security_practices(self) -> QualityGateResult:
        """Validate security practices and potential vulnerabilities."""
        start_time = time.time()
        
        try:
            score = 100.0  # Start with perfect score, deduct for issues
            details = []
            recommendations = []
            security_issues = []
            
            # Security patterns to check
            dangerous_patterns = [
                ("eval(", "Use of eval() function"),
                ("exec(", "Use of exec() function"),
                ("__import__", "Dynamic imports"),
                ("open(", "File operations - review for security"),
                ("subprocess.", "Subprocess usage - verify input sanitization"),
                ("os.system", "Direct system calls"),
                ("pickle.load", "Pickle deserialization - potential RCE")
            ]
            
            safe_patterns = [
                ("logging.", "Logging implementation"),
                ("hashlib.", "Cryptographic hashing"),
                ("secrets.", "Secure random generation"),
                ("try:", "Exception handling")
            ]
            
            python_files = list(self.src_path.glob("*.py"))
            total_files_scanned = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    total_files_scanned += 1
                    
                    # Check for dangerous patterns
                    for pattern, description in dangerous_patterns:
                        if pattern in content:
                            if pattern in ["open(", "subprocess.", "pickle.load"]:
                                score -= 5  # Minor deduction for review items
                                security_issues.append(f"{py_file.name}: {description}")
                            else:
                                score -= 15  # Major deduction for dangerous items
                                security_issues.append(f"{py_file.name}: {description} (HIGH RISK)")
                    
                    # Credit for safe patterns
                    safe_count = sum(1 for pattern, _ in safe_patterns if pattern in content)
                    if safe_count >= 2:
                        score += min(5, safe_count)  # Bonus for good practices
                
                except Exception:
                    continue
            
            details.append(f"Files scanned: {total_files_scanned}")
            details.append(f"Security issues found: {len(security_issues)}")
            
            if security_issues:
                details.extend(security_issues[:5])  # Show first 5 issues
                if len(security_issues) > 5:
                    details.append(f"... and {len(security_issues) - 5} more issues")
                recommendations.append("Review and remediate security issues")
            else:
                details.append("No major security issues detected")
            
            # Check for security documentation
            security_docs = list(self.project_root.glob("*SECURITY*"))
            if security_docs:
                score += 10
                details.append("Security documentation found")
            else:
                recommendations.append("Add security documentation")
            
            status = (QualityGateStatus.PASSED if score >= 80 else 
                     QualityGateStatus.WARNING if score >= 60 else 
                     QualityGateStatus.FAILED)
            
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Security Practices",
                status=status,
                score=max(0, min(100, score)),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Security Practices",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Security scan failed: {e}",
                recommendations=["Fix security analysis errors"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_dependency_security(self) -> QualityGateResult:
        """Validate dependency security and management."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check for requirements/dependency files
            dep_files = [
                "requirements.txt",
                "pyproject.toml", 
                "setup.py",
                "Pipfile"
            ]
            
            found_dep_files = []
            for dep_file in dep_files:
                if (self.project_root / dep_file).exists():
                    found_dep_files.append(dep_file)
                    score += 20
            
            if found_dep_files:
                details.append(f"Dependency files: {', '.join(found_dep_files)}")
            else:
                recommendations.append("Add dependency management files")
            
            # Check pyproject.toml specifically
            pyproject_file = self.project_root / "pyproject.toml"
            if pyproject_file.exists():
                try:
                    with open(pyproject_file, 'r') as f:
                        pyproject_content = f.read()
                    
                    # Check for version constraints
                    if ">=" in pyproject_content:
                        score += 15
                        details.append("Version constraints specified")
                    else:
                        recommendations.append("Add version constraints for dependencies")
                    
                    # Check for optional dependencies
                    if "optional-dependencies" in pyproject_content:
                        score += 15
                        details.append("Optional dependencies defined")
                    
                    # Check for development dependencies
                    if "dev" in pyproject_content or "test" in pyproject_content:
                        score += 15
                        details.append("Development dependencies separated")
                        
                except Exception:
                    recommendations.append("Review pyproject.toml format")
            
            # Check for license
            license_files = list(self.project_root.glob("LICENSE*"))
            if license_files:
                score += 20
                details.append("License file found")
            else:
                recommendations.append("Add license file")
            
            # Check for security-related configs
            security_configs = [".pre-commit-config.yaml", "renovate.json", ".github/dependabot.yml"]
            found_security = [f for f in security_configs if (self.project_root / f).exists()]
            
            if found_security:
                score += 15
                details.append(f"Security automation: {', '.join(found_security)}")
            else:
                recommendations.append("Consider adding automated security checks")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Dependency Security",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Dependency Security",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Dependency analysis failed: {e}",
                recommendations=["Fix dependency security analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_performance_characteristics(self) -> QualityGateResult:
        """Validate performance characteristics and benchmarks."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check for performance-related code patterns
            performance_patterns = [
                ("@dataclass", "Efficient data structures"),
                ("threading", "Concurrent processing"),
                ("concurrent.futures", "Parallel execution"),
                ("cache", "Caching implementation"),
                ("time.time()", "Performance timing"),
                ("@property", "Efficient property access"),
                ("__slots__", "Memory optimization")
            ]
            
            python_files = list(self.src_path.glob("*.py"))
            pattern_matches = {}
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    for pattern, description in performance_patterns:
                        if pattern in content:
                            if description not in pattern_matches:
                                pattern_matches[description] = 0
                            pattern_matches[description] += 1
                            
                except Exception:
                    continue
            
            if pattern_matches:
                score += len(pattern_matches) * 10
                details.append(f"Performance patterns: {', '.join(pattern_matches.keys())}")
            else:
                recommendations.append("Add performance optimization patterns")
            
            # Check for benchmark/profiling files
            perf_files = list(self.project_root.glob("*benchmark*")) + list(self.project_root.glob("*performance*"))
            if perf_files:
                score += 20
                details.append(f"Performance files: {len(perf_files)}")
            else:
                recommendations.append("Add performance benchmarks")
            
            # Look for algorithmic complexity considerations
            complexity_indicators = [
                "O(", "complexity", "algorithm", "optimization", "efficient"
            ]
            
            complexity_mentions = 0
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read().lower()
                    
                    for indicator in complexity_indicators:
                        if indicator in content:
                            complexity_mentions += 1
                            break
                            
                except Exception:
                    continue
            
            if complexity_mentions > 0:
                score += 15
                details.append(f"Algorithmic complexity considerations: {complexity_mentions} files")
            else:
                recommendations.append("Document algorithmic complexity")
            
            # Memory management patterns
            memory_patterns = ["del ", "gc.collect", "weakref", "__slots__"]
            memory_count = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    for pattern in memory_patterns:
                        if pattern in content:
                            memory_count += 1
                            
                except Exception:
                    continue
            
            if memory_count > 0:
                score += 10
                details.append(f"Memory management patterns: {memory_count}")
            
            status = QualityGateStatus.PASSED if score >= 60 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Performance Characteristics",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Performance Characteristics",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Performance analysis failed: {e}",
                recommendations=["Fix performance analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_memory_efficiency(self) -> QualityGateResult:
        """Validate memory efficiency and resource usage."""
        start_time = time.time()
        
        try:
            score = 50.0  # Base score
            details = []
            recommendations = []
            
            # Test basic memory usage by importing modules
            try:
                import tracemalloc
                tracemalloc.start()
                
                # Try to import our modules
                sys.path.append(str(self.src_path.parent))
                
                initial_memory = tracemalloc.get_traced_memory()[0]
                
                # Basic import test (would need dependencies in real environment)
                # For now, just check file sizes as proxy
                
                total_size = 0
                large_files = []
                
                for py_file in self.src_path.glob("*.py"):
                    file_size = py_file.stat().st_size
                    total_size += file_size
                    
                    if file_size > 100000:  # 100KB+
                        large_files.append((py_file.name, file_size))
                
                details.append(f"Total source size: {total_size} bytes")
                
                if total_size < 1000000:  # Under 1MB total
                    score += 20
                    details.append("Reasonable total size")
                else:
                    recommendations.append("Consider optimizing source code size")
                
                if len(large_files) <= 3:  # Max 3 large files
                    score += 15
                else:
                    recommendations.append("Review large file sizes")
                    for name, size in large_files[:3]:
                        details.append(f"Large file: {name} ({size} bytes)")
                
                tracemalloc.stop()
                
                score += 15  # Bonus for completing memory test
                
            except Exception as e:
                details.append(f"Memory analysis limited: {e}")
                recommendations.append("Run in full environment for complete memory analysis")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Memory Efficiency",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Memory Efficiency",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Memory analysis failed: {e}",
                recommendations=["Fix memory efficiency analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_documentation_completeness(self) -> QualityGateResult:
        """Validate documentation completeness and quality."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check for main documentation files
            doc_files = {
                "README.md": 25,
                "CHANGELOG.md": 10,
                "CONTRIBUTING.md": 10,
                "LICENSE": 15,
                "SECURITY.md": 10
            }
            
            found_docs = []
            for doc_file, points in doc_files.items():
                doc_path = self.project_root / doc_file
                if doc_path.exists():
                    score += points
                    file_size = doc_path.stat().st_size
                    found_docs.append(f"{doc_file} ({file_size} bytes)")
                else:
                    recommendations.append(f"Add {doc_file}")
            
            details.append(f"Documentation files: {', '.join(found_docs)}")
            
            # Check README quality
            readme_path = self.project_root / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    readme_content = f.read()
                
                readme_sections = [
                    "# ", "## ", "Installation", "Usage", "Example", 
                    "Features", "License", "Contributing"
                ]
                
                found_sections = sum(1 for section in readme_sections if section.lower() in readme_content.lower())
                if found_sections >= 5:
                    score += 15
                    details.append(f"README sections: {found_sections}")
                else:
                    recommendations.append("Improve README structure")
                
                if len(readme_content) > 5000:  # Substantial README
                    score += 10
                    details.append("Comprehensive README")
            
            # Check for docs directory
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                doc_count = len(list(docs_dir.glob("**/*.md")))
                if doc_count > 5:
                    score += 15
                    details.append(f"Docs directory: {doc_count} files")
                else:
                    recommendations.append("Expand documentation in docs/")
            else:
                recommendations.append("Create docs/ directory")
            
            # Check for research paper
            research_files = list(self.project_root.glob("*RESEARCH*")) + list(self.project_root.glob("*PAPER*"))
            if research_files:
                score += 20
                details.append("Research documentation found")
            else:
                recommendations.append("Add research documentation")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Documentation Completeness",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Documentation Completeness",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Documentation analysis failed: {e}",
                recommendations=["Fix documentation analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_api_documentation(self) -> QualityGateResult:
        """Validate API documentation quality."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            python_files = list(self.src_path.glob("*.py"))
            if not python_files:
                return QualityGateResult(
                    gate_name="API Documentation",
                    status=QualityGateStatus.SKIPPED,
                    score=0.0,
                    details="No Python files found",
                    recommendations=[],
                    execution_time=time.time() - start_time
                )
            
            total_functions = 0
            documented_functions = 0
            total_classes = 0
            documented_classes = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if ast.get_docstring(node):
                                documented_classes += 1
                                
                except Exception:
                    continue
            
            # Calculate documentation coverage
            func_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
            class_coverage = (documented_classes / total_classes * 100) if total_classes > 0 else 100
            
            if func_coverage >= 80:
                score += 40
            elif func_coverage >= 60:
                score += 25
            elif func_coverage >= 40:
                score += 15
            else:
                recommendations.append("Improve function documentation coverage")
            
            if class_coverage >= 80:
                score += 30
            elif class_coverage >= 60:
                score += 20
            else:
                recommendations.append("Improve class documentation coverage")
            
            details.append(f"Function docs: {func_coverage:.1f}% ({documented_functions}/{total_functions})")
            details.append(f"Class docs: {class_coverage:.1f}% ({documented_classes}/{total_classes})")
            
            # Check for type hints
            type_hint_files = 0
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    if "typing" in content or "->" in content or ": " in content:
                        type_hint_files += 1
                        
                except Exception:
                    continue
            
            if type_hint_files >= len(python_files) * 0.8:  # 80% of files
                score += 20
                details.append(f"Type hints: {type_hint_files}/{len(python_files)} files")
            else:
                recommendations.append("Add type hints to more files")
            
            # Check for docstring quality indicators
            quality_indicators = ["Args:", "Returns:", "Raises:", "Example:", "Note:"]
            quality_score = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    for indicator in quality_indicators:
                        if indicator in content:
                            quality_score += 1
                            
                except Exception:
                    continue
            
            if quality_score >= 3:
                score += 10
                details.append("Quality docstring patterns found")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="API Documentation",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="API Documentation",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"API documentation analysis failed: {e}",
                recommendations=["Fix API documentation analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_research_reproducibility(self) -> QualityGateResult:
        """Validate research reproducibility and experimental setup."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check for research-related files
            research_indicators = [
                ("*research*", "Research modules"),
                ("*experiment*", "Experimental framework"),
                ("*benchmark*", "Benchmarking code"),
                ("*validation*", "Validation framework"),
                ("*test*", "Testing infrastructure")
            ]
            
            found_research = {}
            for pattern, description in research_indicators:
                matches = list(self.project_root.glob(f"**/{pattern}.py"))
                if matches:
                    found_research[description] = len(matches)
                    score += 15
            
            if found_research:
                details.append(f"Research components: {', '.join(found_research.keys())}")
            else:
                recommendations.append("Add research and experimental components")
            
            # Check for research paper/documentation
            research_docs = (list(self.project_root.glob("*RESEARCH*")) + 
                           list(self.project_root.glob("*PAPER*")) + 
                           list(self.project_root.glob("*PUBLICATION*")))
            
            if research_docs:
                score += 25
                details.append(f"Research documents: {len(research_docs)}")
            else:
                recommendations.append("Add research documentation")
            
            # Check for reproducibility features
            repro_patterns = [
                ("random.seed", "Random seed control"),
                ("np.random.seed", "NumPy seed control"), 
                ("config", "Configuration management"),
                ("experiment_id", "Experiment tracking"),
                ("timestamp", "Temporal tracking")
            ]
            
            repro_features = []
            python_files = list(self.project_root.glob("**/*.py"))
            
            for pattern, description in repro_patterns:
                found = any(pattern in py_file.read_text(errors='ignore') 
                          for py_file in python_files if py_file.is_file())
                if found:
                    repro_features.append(description)
                    score += 8
            
            if repro_features:
                details.append(f"Reproducibility: {', '.join(repro_features)}")
            else:
                recommendations.append("Add reproducibility controls")
            
            # Check for statistical analysis
            stats_indicators = ["statistics", "scipy", "significance", "p_value", "confidence"]
            stats_found = []
            
            for py_file in python_files:
                try:
                    content = py_file.read_text(errors='ignore')
                    for indicator in stats_indicators:
                        if indicator in content.lower():
                            stats_found.append(indicator)
                            break
                except Exception:
                    continue
            
            if stats_found:
                score += 15
                details.append(f"Statistical analysis: {len(stats_found)} indicators")
            else:
                recommendations.append("Add statistical analysis components")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Research Reproducibility",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Research Reproducibility",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Reproducibility analysis failed: {e}",
                recommendations=["Fix reproducibility analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_experimental_framework(self) -> QualityGateResult:
        """Validate experimental framework completeness."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check for breakthrough research files specifically
            breakthrough_files = [
                "meta_autonomous_evolution_engine.py",
                "breakthrough_research_framework.py"
            ]
            
            found_breakthrough = []
            for file_name in breakthrough_files:
                file_path = self.src_path / file_name
                if file_path.exists():
                    found_breakthrough.append(file_name)
                    score += 30
                    
                    # Check file size (should be substantial)
                    file_size = file_path.stat().st_size
                    if file_size > 10000:  # 10KB+
                        score += 10
                        details.append(f"{file_name}: {file_size} bytes")
            
            if found_breakthrough:
                details.append(f"Breakthrough components: {', '.join(found_breakthrough)}")
            else:
                recommendations.append("Add breakthrough research components")
            
            # Check for integration tests
            integration_tests = list(self.project_root.glob("*integration*"))
            if integration_tests:
                score += 20
                details.append(f"Integration tests: {len(integration_tests)}")
            else:
                recommendations.append("Add integration test suite")
            
            # Check for demonstration functions
            demo_patterns = ["demonstrate", "example", "showcase", "demo"]
            demo_found = 0
            
            python_files = list(self.src_path.glob("*.py"))
            for py_file in python_files:
                try:
                    content = py_file.read_text(errors='ignore')
                    for pattern in demo_patterns:
                        if f"def {pattern}" in content.lower():
                            demo_found += 1
                            break
                except Exception:
                    continue
            
            if demo_found > 0:
                score += 15
                details.append(f"Demonstration functions: {demo_found}")
            else:
                recommendations.append("Add demonstration functions")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING  
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Experimental Framework",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Experimental Framework",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Framework analysis failed: {e}",
                recommendations=["Fix experimental framework analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_production_readiness(self) -> QualityGateResult:
        """Validate production deployment readiness."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check for deployment files
            deployment_files = {
                "Dockerfile": 15,
                "docker-compose.yml": 10,
                "requirements.txt": 15,
                "pyproject.toml": 20,
                ".gitignore": 10
            }
            
            found_deployment = []
            for deploy_file, points in deployment_files.items():
                if (self.project_root / deploy_file).exists():
                    score += points
                    found_deployment.append(deploy_file)
                else:
                    recommendations.append(f"Add {deploy_file}")
            
            details.append(f"Deployment files: {', '.join(found_deployment)}")
            
            # Check for CI/CD configuration
            ci_patterns = [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile", ".circleci"]
            ci_found = []
            
            for pattern in ci_patterns:
                if (self.project_root / pattern).exists():
                    ci_found.append(pattern)
                    score += 10
            
            if ci_found:
                details.append(f"CI/CD: {', '.join(ci_found)}")
            else:
                recommendations.append("Add CI/CD configuration")
            
            # Check for monitoring/observability
            monitoring_files = ["prometheus", "grafana", "health", "metrics", "logging"]
            monitoring_found = []
            
            all_files = list(self.project_root.glob("**/*"))
            for monitor_pattern in monitoring_files:
                if any(monitor_pattern in str(f).lower() for f in all_files):
                    monitoring_found.append(monitor_pattern)
                    score += 5
            
            if monitoring_found:
                details.append(f"Monitoring: {', '.join(monitoring_found)}")
            else:
                recommendations.append("Add monitoring and observability")
            
            # Check for production configurations
            config_files = list(self.project_root.glob("config/*")) + list(self.project_root.glob("*config*"))
            if config_files:
                score += 15
                details.append(f"Configuration files: {len(config_files)}")
            else:
                recommendations.append("Add production configuration")
            
            status = QualityGateStatus.PASSED if score >= 70 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Production Readiness",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Production Readiness",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Production analysis failed: {e}",
                recommendations=["Fix production readiness analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def validate_configuration_management(self) -> QualityGateResult:
        """Validate configuration management practices."""
        start_time = time.time()
        
        try:
            score = 0.0
            details = []
            recommendations = []
            
            # Check for configuration structure
            config_locations = [
                ("config/", "Dedicated config directory"),
                ("src/*/config", "Package-level config"),
                ("*.env", "Environment files"),
                ("settings.py", "Settings module")
            ]
            
            found_configs = []
            for location, description in config_locations:
                if list(self.project_root.glob(location)):
                    found_configs.append(description)
                    score += 20
            
            if found_configs:
                details.append(f"Config structure: {', '.join(found_configs)}")
            else:
                recommendations.append("Add configuration management structure")
            
            # Check for environment separation
            env_indicators = ["production", "development", "test", "staging"]
            env_files = []
            
            all_files = list(self.project_root.glob("**/*"))
            for env in env_indicators:
                env_matches = [f for f in all_files if env in str(f).lower()]
                if env_matches:
                    env_files.append(env)
                    score += 10
            
            if env_files:
                details.append(f"Environment separation: {', '.join(env_files)}")
            else:
                recommendations.append("Add environment-specific configurations")
            
            # Check for secrets management
            secrets_patterns = ["secret", "key", "token", "password"]
            secrets_handling = []
            
            python_files = list(self.src_path.glob("*.py"))
            for py_file in python_files:
                try:
                    content = py_file.read_text(errors='ignore')
                    if "os.environ" in content or "getenv" in content:
                        secrets_handling.append("Environment variables")
                        score += 15
                        break
                except Exception:
                    continue
            
            if secrets_handling:
                details.append(f"Secrets handling: {', '.join(secrets_handling)}")
            else:
                recommendations.append("Add proper secrets management")
            
            status = QualityGateStatus.PASSED if score >= 60 else QualityGateStatus.WARNING
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Configuration Management",
                status=status,
                score=min(100, score),
                details="; ".join(details),
                recommendations=recommendations,
                execution_time=execution_time
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Configuration Management",
                status=QualityGateStatus.FAILED,
                score=0.0,
                details=f"Config analysis failed: {e}",
                recommendations=["Fix configuration management analysis"],
                execution_time=time.time() - start_time
            )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def _print_result(self, result: QualityGateResult):
        """Print quality gate result."""
        print(f"{result.status.value} {result.gate_name}")
        print(f"    Score: {result.score:.1f}/100")
        print(f"    Time: {result.execution_time:.3f}s")
        if result.details:
            print(f"    Details: {result.details}")
        if result.recommendations:
            print(f"    Recommendations: {', '.join(result.recommendations[:2])}")
        print()
    
    def _generate_quality_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        # Calculate overall metrics
        total_results = len(self.results)
        passed_count = sum(1 for r in self.results if r.status == QualityGateStatus.PASSED)
        warning_count = sum(1 for r in self.results if r.status == QualityGateStatus.WARNING) 
        failed_count = sum(1 for r in self.results if r.status == QualityGateStatus.FAILED)
        skipped_count = sum(1 for r in self.results if r.status == QualityGateStatus.SKIPPED)
        
        # Calculate average score
        scored_results = [r for r in self.results if r.score > 0]
        avg_score = sum(r.score for r in scored_results) / len(scored_results) if scored_results else 0
        
        # Determine overall status
        if failed_count > 0:
            overall_status = QualityGateStatus.FAILED
        elif warning_count > total_results * 0.3:  # More than 30% warnings
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.PASSED
        
        # Generate summary
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": overall_status.value,
            "summary": {
                "total_gates": total_results,
                "passed": passed_count,
                "warnings": warning_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "average_score": round(avg_score, 1),
                "execution_time": round(total_time, 2)
            },
            "gate_results": [
                {
                    "name": r.gate_name,
                    "status": r.status.value,
                    "score": r.score,
                    "details": r.details,
                    "recommendations": r.recommendations,
                    "execution_time": r.execution_time
                }
                for r in self.results
            ],
            "top_recommendations": self._get_top_recommendations(),
            "production_readiness": self._assess_production_readiness(avg_score, failed_count),
            "research_validation": self._assess_research_validation()
        }
        
        # Print final summary
        print("=" * 60)
        print(f"{overall_status.value} QUALITY GATES VALIDATION COMPLETE")
        print("=" * 60)
        print(f"📊 Summary: {passed_count} passed, {warning_count} warnings, {failed_count} failed")
        print(f"🎯 Average Score: {avg_score:.1f}/100")
        print(f"⏱️ Total Time: {total_time:.2f}s")
        print(f"🚀 Production Ready: {report['production_readiness']['ready']}")
        print(f"🔬 Research Validated: {report['research_validation']['validated']}")
        
        return report
    
    def _get_top_recommendations(self) -> List[str]:
        """Get top recommendations across all gates."""
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        # Count frequency of similar recommendations
        recommendation_counts = {}
        for rec in all_recommendations:
            key = rec.lower()[:30]  # First 30 chars as key
            if key not in recommendation_counts:
                recommendation_counts[key] = (rec, 0)
            recommendation_counts[key] = (recommendation_counts[key][0], recommendation_counts[key][1] + 1)
        
        # Sort by frequency and return top 5
        sorted_recs = sorted(recommendation_counts.values(), key=lambda x: x[1], reverse=True)
        return [rec[0] for rec in sorted_recs[:5]]
    
    def _assess_production_readiness(self, avg_score: float, failed_count: int) -> Dict[str, Any]:
        """Assess overall production readiness."""
        ready = avg_score >= 75 and failed_count == 0
        
        readiness_factors = {
            "minimum_score_met": avg_score >= 75,
            "no_critical_failures": failed_count == 0,
            "security_validated": any(r.gate_name == "Security Practices" and r.status == QualityGateStatus.PASSED for r in self.results),
            "documentation_complete": any(r.gate_name == "Documentation Completeness" and r.score >= 70 for r in self.results),
            "performance_validated": any(r.gate_name == "Performance Characteristics" and r.score >= 60 for r in self.results)
        }
        
        return {
            "ready": ready,
            "score": avg_score,
            "factors": readiness_factors,
            "blocking_issues": failed_count
        }
    
    def _assess_research_validation(self) -> Dict[str, Any]:
        """Assess research validation readiness."""
        research_gates = [r for r in self.results if "Research" in r.gate_name or "Experimental" in r.gate_name]
        
        validated = len(research_gates) > 0 and all(r.score >= 70 for r in research_gates)
        
        research_components = {
            "experimental_framework": any("Experimental" in r.gate_name and r.score >= 70 for r in research_gates),
            "reproducibility": any("Reproducibility" in r.gate_name and r.score >= 70 for r in research_gates),
            "documentation": any(r.gate_name == "Documentation Completeness" and r.score >= 80 for r in self.results)
        }
        
        return {
            "validated": validated,
            "components": research_components,
            "research_gates_passed": len([r for r in research_gates if r.status == QualityGateStatus.PASSED])
        }


def main():
    """Run quality gates validation."""
    validator = QualityGatesValidator()
    report = validator.run_all_quality_gates()
    
    # Save report
    report_file = "quality_gates_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Full report saved to: {report_file}")
    
    # Return exit code based on results
    if report["overall_status"] == QualityGateStatus.FAILED.value:
        return 1
    elif report["overall_status"] == QualityGateStatus.WARNING.value:
        return 2
    else:
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)