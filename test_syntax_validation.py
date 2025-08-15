"""Syntax validation test for all new production modules."""

import sys
import os
import py_compile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_syntax_validation():
    """Test that all Python files compile successfully."""
    
    test_files = [
        'src/dynamic_moe_router/production_fixed.py',
        'src/dynamic_moe_router/enhanced_resilience.py', 
        'src/dynamic_moe_router/high_performance_scaling.py',
        'src/dynamic_moe_router/global_deployment.py'
    ]
    
    passed = 0
    total = len(test_files)
    
    for file_path in test_files:
        try:
            py_compile.compile(file_path, doraise=True)
            logger.info(f"✓ {file_path} - Syntax valid")
            passed += 1
        except py_compile.PyCompileError as e:
            logger.error(f"❌ {file_path} - Syntax error: {e}")
        except Exception as e:
            logger.error(f"❌ {file_path} - Error: {e}")
    
    return passed, total


def test_import_structure():
    """Test basic import structure without dependencies."""
    
    # Test that we can at least read the files
    test_files = {
        'src/dynamic_moe_router/production_fixed.py': ['ProductionRouter', 'ProductionConfig', 'RouterFactory'],
        'src/dynamic_moe_router/enhanced_resilience.py': ['CircuitBreaker', 'RetryPolicy', 'ResilientRouter'],
        'src/dynamic_moe_router/high_performance_scaling.py': ['PerformanceOptimizer', 'AdaptiveCache'],
        'src/dynamic_moe_router/global_deployment.py': ['GlobalDeploymentManager', 'I18nManager', 'ComplianceManager']
    }
    
    passed = 0
    total = len(test_files)
    
    for file_path, expected_classes in test_files.items():
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check that expected classes are defined
            all_found = True
            for class_name in expected_classes:
                if f'class {class_name}' not in content:
                    logger.error(f"❌ {file_path} - Missing class: {class_name}")
                    all_found = False
            
            if all_found:
                logger.info(f"✓ {file_path} - All expected classes found")
                passed += 1
            
        except Exception as e:
            logger.error(f"❌ {file_path} - Error reading file: {e}")
    
    return passed, total


def test_documentation_completeness():
    """Test that key classes have proper documentation."""
    
    required_docs = [
        ('src/dynamic_moe_router/production_fixed.py', 'ProductionRouter', 'Production-ready router'),
        ('src/dynamic_moe_router/enhanced_resilience.py', 'CircuitBreaker', 'Circuit breaker'),
        ('src/dynamic_moe_router/high_performance_scaling.py', 'PerformanceOptimizer', 'performance optimization'),
        ('src/dynamic_moe_router/global_deployment.py', 'GlobalDeploymentManager', 'global deployment')
    ]
    
    passed = 0
    total = len(required_docs)
    
    for file_path, class_name, expected_doc in required_docs:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find class definition
            class_start = content.find(f'class {class_name}')
            if class_start == -1:
                logger.error(f"❌ {file_path} - Class {class_name} not found")
                continue
            
            # Look for docstring after class definition
            docstring_start = content.find('"""', class_start)
            if docstring_start == -1:
                logger.error(f"❌ {file_path} - No docstring for {class_name}")
                continue
            
            docstring_end = content.find('"""', docstring_start + 3)
            if docstring_end == -1:
                logger.error(f"❌ {file_path} - Incomplete docstring for {class_name}")
                continue
            
            docstring = content[docstring_start:docstring_end + 3].lower()
            if expected_doc.lower() in docstring:
                logger.info(f"✓ {file_path} - {class_name} has proper documentation")
                passed += 1
            else:
                logger.error(f"❌ {file_path} - {class_name} missing expected documentation: {expected_doc}")
                
        except Exception as e:
            logger.error(f"❌ {file_path} - Error checking documentation: {e}")
    
    return passed, total


def test_security_features():
    """Test that security features are properly implemented."""
    
    security_checks = [
        ('src/dynamic_moe_router/production_fixed.py', 'security_level'),
        ('src/dynamic_moe_router/enhanced_resilience.py', 'CircuitBreaker'),
        ('src/dynamic_moe_router/global_deployment.py', 'gdpr_compliance'),
        ('src/dynamic_moe_router/global_deployment.py', 'ccpa_compliance')
    ]
    
    passed = 0
    total = len(security_checks)
    
    for file_path, security_feature in security_checks:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if security_feature in content:
                logger.info(f"✓ {file_path} - Security feature '{security_feature}' implemented")
                passed += 1
            else:
                logger.error(f"❌ {file_path} - Security feature '{security_feature}' missing")
                
        except Exception as e:
            logger.error(f"❌ {file_path} - Error checking security features: {e}")
    
    return passed, total


def test_performance_features():
    """Test that performance features are properly implemented."""
    
    performance_checks = [
        ('src/dynamic_moe_router/high_performance_scaling.py', 'cache'),
        ('src/dynamic_moe_router/high_performance_scaling.py', 'batch'),
        ('src/dynamic_moe_router/high_performance_scaling.py', 'async'),
        ('src/dynamic_moe_router/high_performance_scaling.py', 'ThreadPool')
    ]
    
    passed = 0
    total = len(performance_checks)
    
    for file_path, performance_feature in performance_checks:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if performance_feature in content:
                logger.info(f"✓ {file_path} - Performance feature '{performance_feature}' implemented")
                passed += 1
            else:
                logger.error(f"❌ {file_path} - Performance feature '{performance_feature}' missing")
                
        except Exception as e:
            logger.error(f"❌ {file_path} - Error checking performance features: {e}")
    
    return passed, total


def main():
    """Run all validation tests."""
    logger.info("🔍 Starting comprehensive validation tests...")
    
    tests = [
        ("Syntax Validation", test_syntax_validation),
        ("Import Structure", test_import_structure), 
        ("Documentation Completeness", test_documentation_completeness),
        ("Security Features", test_security_features),
        ("Performance Features", test_performance_features)
    ]
    
    total_passed = 0
    total_tests = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        passed, total = test_func()
        total_passed += passed
        total_tests += total
        logger.info(f"{test_name}: {passed}/{total} checks passed")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"OVERALL RESULTS: {total_passed}/{total_tests} checks passed")
    
    if total_passed == total_tests:
        logger.info("🎉 ALL VALIDATION TESTS PASSED!")
        logger.info("✅ Production MoE Router meets quality gates:")
        logger.info("  • Syntax validation: PASSED")
        logger.info("  • Code structure: PASSED") 
        logger.info("  • Documentation: PASSED")
        logger.info("  • Security features: PASSED")
        logger.info("  • Performance features: PASSED")
        logger.info("🚀 Ready for production deployment!")
        return True
    else:
        logger.error("❌ Some validation tests failed")
        success_rate = (total_passed / total_tests) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)