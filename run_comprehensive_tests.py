#!/usr/bin/env python3
"""Run comprehensive test suite for autonomous SDLC quality system."""

import os
import sys
import time
import subprocess
import unittest
from pathlib import Path
from typing import Dict, List, Any

# Add source paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 COMPREHENSIVE TEST SUITE - AUTONOMOUS SDLC QUALITY")
print("=" * 70)

def run_command(cmd: str, description: str) -> Dict[str, Any]:
    """Run a command and capture results."""
    print(f"⚡ {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        execution_time = time.time() - start_time
        
        return {
            "command": cmd,
            "description": description,
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": execution_time
        }
        
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "description": description,
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": "Command timed out",
            "execution_time": time.time() - start_time
        }
    except Exception as e:
        return {
            "command": cmd,
            "description": description,
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time
        }

def run_python_tests() -> List[Dict[str, Any]]:
    """Run Python-based tests."""
    test_results = []
    
    # Test quality gates validation
    result = run_command(
        "python3 quality_gates_validation.py",
        "Quality Gates Validation"
    )
    test_results.append(result)
    
    # Test autonomous quality assessment
    result = run_command(
        "python3 run_autonomous_quality_assessment.py",
        "Autonomous Quality Assessment"
    )
    test_results.append(result)
    
    return test_results

def run_syntax_validation() -> List[Dict[str, Any]]:
    """Run syntax validation on all Python files."""
    test_results = []
    
    # Find all Python files
    python_files = []
    for root in ["src", "."]:
        if os.path.exists(root):
            for path in Path(root).rglob("*.py"):
                python_files.append(str(path))
    
    print(f"📝 Found {len(python_files)} Python files to validate")
    
    # Check syntax for each file
    failed_files = []
    total_files = len(python_files)
    
    for i, py_file in enumerate(python_files):
        if i % 10 == 0:  # Progress indicator
            print(f"  Validating files: {i+1}/{total_files}")
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile
            compile(content, py_file, 'exec')
            
        except SyntaxError as e:
            failed_files.append({
                "file": py_file,
                "error": f"Syntax error: {e}",
                "line": getattr(e, 'lineno', 0)
            })
        except Exception as e:
            failed_files.append({
                "file": py_file,
                "error": f"Parse error: {e}",
                "line": 0
            })
    
    result = {
        "command": "syntax_validation",
        "description": "Python Syntax Validation",
        "success": len(failed_files) == 0,
        "return_code": 0 if len(failed_files) == 0 else 1,
        "stdout": f"Validated {total_files} files, {len(failed_files)} failures",
        "stderr": "\n".join([f"{f['file']}:{f['line']} - {f['error']}" for f in failed_files]),
        "execution_time": 0.0,
        "files_checked": total_files,
        "failures": failed_files
    }
    
    test_results.append(result)
    return test_results

def run_import_validation() -> List[Dict[str, Any]]:
    """Validate that key modules can be imported."""
    test_results = []
    
    key_modules = [
        "quality_gates_validation",
        "autonomous_sdlc_quality_integration",
        "src.dynamic_moe_router"
    ]
    
    for module in key_modules:
        try:
            start_time = time.time()
            
            if module == "src.dynamic_moe_router":
                # Special handling for package import
                import src.dynamic_moe_router
                success = True
                error = ""
            else:
                __import__(module)
                success = True
                error = ""
            
            execution_time = time.time() - start_time
            
            result = {
                "command": f"import {module}",
                "description": f"Import {module}",
                "success": success,
                "return_code": 0,
                "stdout": f"Successfully imported {module}",
                "stderr": error,
                "execution_time": execution_time
            }
            
        except Exception as e:
            result = {
                "command": f"import {module}",
                "description": f"Import {module}",
                "success": False,
                "return_code": 1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": time.time() - start_time
            }
        
        test_results.append(result)
    
    return test_results

def run_performance_benchmarks() -> List[Dict[str, Any]]:
    """Run performance benchmarks."""
    test_results = []
    
    # Benchmark quality gates validation speed
    start_time = time.time()
    try:
        from quality_gates_validation import QualityGatesValidator
        validator = QualityGatesValidator()
        
        # Run 3 times and take average
        times = []
        for _ in range(3):
            run_start = time.time()
            validator.run_all_quality_gates()
            times.append(time.time() - run_start)
        
        avg_time = sum(times) / len(times)
        
        result = {
            "command": "performance_benchmark",
            "description": "Quality Gates Performance Benchmark",
            "success": avg_time < 10.0,  # Should complete in under 10 seconds
            "return_code": 0 if avg_time < 10.0 else 1,
            "stdout": f"Average execution time: {avg_time:.2f}s (3 runs)",
            "stderr": "" if avg_time < 10.0 else "Performance benchmark failed - too slow",
            "execution_time": time.time() - start_time,
            "benchmark_times": times,
            "average_time": avg_time
        }
        
    except Exception as e:
        result = {
            "command": "performance_benchmark",
            "description": "Quality Gates Performance Benchmark", 
            "success": False,
            "return_code": 1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time
        }
    
    test_results.append(result)
    return test_results

def calculate_test_coverage(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate test coverage metrics."""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result["success"])
    failed_tests = total_tests - passed_tests
    
    coverage_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "coverage_percentage": coverage_percentage,
        "overall_success": failed_tests == 0
    }

def generate_test_report(all_results: List[Dict[str, Any]], coverage: Dict[str, Any]) -> str:
    """Generate comprehensive test report."""
    report = f"""# Comprehensive Test Suite Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Test Coverage Summary

- **Total Tests:** {coverage['total_tests']}
- **Passed:** {coverage['passed_tests']} ✅
- **Failed:** {coverage['failed_tests']} ❌
- **Success Rate:** {coverage['coverage_percentage']:.1f}%
- **Overall Status:** {'PASSED' if coverage['overall_success'] else 'FAILED'}

## Test Results

"""
    
    for result in all_results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        report += f"### {status} {result['description']}\n\n"
        report += f"- **Command:** `{result['command']}`\n"
        report += f"- **Return Code:** {result['return_code']}\n"
        report += f"- **Execution Time:** {result['execution_time']:.2f}s\n"
        
        if result["stdout"]:
            report += f"- **Output:** {result['stdout'][:200]}...\n"
        
        if result["stderr"]:
            report += f"- **Error:** {result['stderr'][:200]}...\n"
        
        report += "\n"
    
    # Add recommendations
    if coverage['failed_tests'] > 0:
        report += "## Recommendations\n\n"
        for result in all_results:
            if not result["success"]:
                report += f"- Fix {result['description']}: {result['stderr'][:100]}\n"
    
    return report

def main():
    """Run comprehensive test suite."""
    all_test_results = []
    
    print("🔍 Phase 1: Python Syntax Validation")
    syntax_results = run_syntax_validation()
    all_test_results.extend(syntax_results)
    
    print("\n🔗 Phase 2: Import Validation") 
    import_results = run_import_validation()
    all_test_results.extend(import_results)
    
    print("\n🧪 Phase 3: Functional Tests")
    python_results = run_python_tests()
    all_test_results.extend(python_results)
    
    print("\n⚡ Phase 4: Performance Benchmarks")
    perf_results = run_performance_benchmarks()
    all_test_results.extend(perf_results)
    
    # Calculate coverage
    coverage = calculate_test_coverage(all_test_results)
    
    # Display summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {coverage['total_tests']}")
    print(f"Passed: {coverage['passed_tests']} ✅")
    print(f"Failed: {coverage['failed_tests']} ❌") 
    print(f"Success Rate: {coverage['coverage_percentage']:.1f}%")
    print(f"Overall Status: {'PASSED ✅' if coverage['overall_success'] else 'FAILED ❌'}")
    
    # Show failed tests
    if coverage['failed_tests'] > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in all_test_results:
            if not result["success"]:
                print(f"  • {result['description']}: {result['stderr'][:100]}")
    
    # Generate and save report
    report_content = generate_test_report(all_test_results, coverage)
    report_file = f"comprehensive_test_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    # Return appropriate exit code
    return 0 if coverage['overall_success'] else 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n🎉 Test suite completed with exit code: {exit_code}")
    sys.exit(exit_code)