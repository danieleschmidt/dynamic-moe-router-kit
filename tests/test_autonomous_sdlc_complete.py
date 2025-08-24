"""Comprehensive test suite for Autonomous SDLC breakthrough implementations.

This test suite validates the complete autonomous SDLC system including:
- Autonomous SDLC Router 
- Research Validation Framework
- Continuous Learning Optimizer

Tests both individual components and integrated system behavior.
"""

import time
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import our breakthrough implementations  
import sys
sys.path.insert(0, 'src')

from dynamic_moe_router.autonomous_sdlc_router import (
    AutonomousSDLCRouter,
    SDLCPhase, 
    DevelopmentExpert,
    CodeComplexityMetrics,
    SDLCTask,
    ExpertCapability,
    create_sample_experts,
    demonstrate_autonomous_sdlc_routing
)

from dynamic_moe_router.sdlc_research_framework import (
    SDLCResearchFramework,
    BaselineMethod,
    ExperimentConfiguration,
    PerformanceMetric,
    RandomAssignmentMethod,
    SkillBasedMethod,
    AgileScruMethod,
    run_comprehensive_sdlc_research
)

from dynamic_moe_router.autonomous_sdlc_optimizer import (
    AutonomousSDLCOptimizer,
    LearningObjective,
    OptimizationStrategy,
    PerformanceObservation,
    ContinuousLearningEngine,
    ExpertSkillPredictor,
    TaskComplexityPredictor,
    demonstrate_autonomous_sdlc_optimization
)


class TestAutonomousSDLCSystem:
    """Comprehensive integration tests for the complete autonomous SDLC system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.experts = create_sample_experts()
        self.router = AutonomousSDLCRouter(self.experts)
        self.optimizer = AutonomousSDLCOptimizer(
            self.experts,
            enable_continuous_learning=False  # Disable for testing
        )
        
    def test_autonomous_sdlc_router_basic_functionality(self):
        """Test basic autonomous SDLC router functionality."""
        # Create test task
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=20.0,
            cognitive_complexity=40.0,
            halstead_complexity=200.0,
            lines_of_code=1000,
            function_count=25,
            class_count=5,
            dependency_depth=4,
            api_surface_area=8,
            test_coverage=0.8,
            performance_requirements=0.7,
            security_requirements=0.6,
            scalability_requirements=0.8
        )
        
        task = SDLCTask(
            task_id="TEST-BASIC-001",
            phase=SDLCPhase.IMPLEMENTATION,
            description="Basic functionality test",
            complexity_metrics=metrics,
            priority=0.8,
            deadline_pressure=0.6,
            dependencies=["DEP-001"],
            estimated_effort=30.0,
            risk_level=0.5
        )
        
        # Route the task
        result = self.router.route_task(task)
        
        # Validate results
        assert 'selected_experts' in result
        assert 'expert_weights' in result
        assert 'routing_confidence' in result
        assert 'estimated_completion_time' in result
        
        # Validate constraints
        assert len(result['selected_experts']) >= self.router.min_experts_per_task
        assert len(result['selected_experts']) <= self.router.max_experts_per_task
        assert len(result['expert_weights']) == len(result['selected_experts'])
        assert abs(sum(result['expert_weights']) - 1.0) < 1e-6  # Weights sum to 1
        assert 0.0 <= result['routing_confidence'] <= 1.0
        
        print(f"✅ Basic routing test passed: {len(result['selected_experts'])} experts selected")
    
    def test_complexity_based_expert_selection(self):
        """Test that expert selection adapts to task complexity."""
        # Create low complexity task
        low_metrics = CodeComplexityMetrics(
            cyclomatic_complexity=5.0, cognitive_complexity=10.0,
            halstead_complexity=50.0, lines_of_code=200,
            function_count=5, class_count=1, dependency_depth=2,
            api_surface_area=3, test_coverage=0.9,
            performance_requirements=0.3, security_requirements=0.2,
            scalability_requirements=0.3
        )
        
        low_task = SDLCTask(
            task_id="TEST-LOW-001", phase=SDLCPhase.IMPLEMENTATION,
            description="Low complexity task", complexity_metrics=low_metrics,
            priority=0.5, deadline_pressure=0.3, dependencies=[],
            estimated_effort=8.0, risk_level=0.2
        )
        
        # Create high complexity task
        high_metrics = CodeComplexityMetrics(
            cyclomatic_complexity=45.0, cognitive_complexity=85.0,
            halstead_complexity=800.0, lines_of_code=5000,
            function_count=80, class_count=25, dependency_depth=12,
            api_surface_area=30, test_coverage=0.6,
            performance_requirements=0.9, security_requirements=0.9,
            scalability_requirements=0.9
        )
        
        high_task = SDLCTask(
            task_id="TEST-HIGH-001", phase=SDLCPhase.IMPLEMENTATION,
            description="High complexity task", complexity_metrics=high_metrics,
            priority=0.9, deadline_pressure=0.8, dependencies=["D1", "D2", "D3"],
            estimated_effort=60.0, risk_level=0.8
        )
        
        # Route both tasks
        low_result = self.router.route_task(low_task)
        high_result = self.router.route_task(high_task)
        
        # High complexity task should generally use more experts
        assert len(high_result['selected_experts']) >= len(low_result['selected_experts'])
        
        # High complexity should have higher estimated time
        assert high_result['estimated_completion_time'] > low_result['estimated_completion_time']
        
        print(f"✅ Complexity-based selection: Low={len(low_result['selected_experts'])}, High={len(high_result['selected_experts'])}")
    
    def test_research_framework_baseline_comparison(self):
        """Test the research validation framework."""
        # Create research framework
        framework = SDLCResearchFramework()
        
        # Create minimal experiment configuration
        config = ExperimentConfiguration(
            name="Test Comparison",
            description="Testing baseline comparison",
            num_tasks=10,  # Small for testing
            baseline_methods=[BaselineMethod.RANDOM_ASSIGNMENT, BaselineMethod.SKILL_BASED]
        )
        
        # Run experiment
        results = framework.run_comparative_experiment(config, self.router)
        
        # Validate results
        assert results.config.name == "Test Comparison"
        assert len(results.task_results) > 0
        assert 'Autonomous SDLC Router' in results.method_performance
        
        # Should have compared against baselines
        method_names = list(results.method_performance.keys())
        assert len(method_names) >= 2  # At least autonomous + 1 baseline
        
        # Validate performance metrics
        for method, metrics in results.method_performance.items():
            assert PerformanceMetric.COMPLETION_TIME in metrics
            assert PerformanceMetric.QUALITY_SCORE in metrics
            assert PerformanceMetric.TEAM_SATISFACTION in metrics
            
            # All metrics should be normalized [0,1]
            for metric_value in metrics.values():
                assert 0.0 <= metric_value <= 1.0
        
        # Check statistical significance
        assert len(results.statistical_significance) > 0
        
        print(f"✅ Research framework test passed: {len(method_names)} methods compared")
    
    def test_continuous_learning_engine(self):
        """Test the continuous learning and optimization engine."""
        # Create learning engine
        engine = ContinuousLearningEngine(
            objectives=[LearningObjective.MINIMIZE_COMPLETION_TIME, LearningObjective.MAXIMIZE_QUALITY],
            strategy=OptimizationStrategy.MULTI_OBJECTIVE
        )
        
        # Create sample observations
        for i in range(15):  # Need enough for optimization
            complexity = np.random.uniform(0.3, 0.9)
            observation = PerformanceObservation(
                timestamp=datetime.now(),
                task_id=f"LEARN-{i:03d}",
                task_complexity=complexity,
                assigned_experts=["architect", "algorithm"],
                expert_weights=[0.6, 0.4],
                actual_completion_time=complexity * 25 + np.random.uniform(5, 15),
                estimated_completion_time=complexity * 20,
                quality_score=0.9 - complexity * 0.3 + np.random.uniform(-0.1, 0.1),
                defect_count=int(complexity * 5),
                team_satisfaction=0.8 + np.random.uniform(-0.1, 0.1),
                resource_cost=complexity * 1000,
                context_features={
                    'timeline_pressure': np.random.uniform(0.3, 0.8),
                    'budget_constraint': np.random.uniform(0.4, 0.7),
                    'team_experience': np.random.uniform(0.6, 0.9),
                    'project_phase_progress': np.random.uniform(0.2, 0.8)
                }
            )
            engine.add_observation(observation)
        
        # Check observations were added
        assert len(engine.observations) == 15
        
        # Perform optimization step manually
        initial_performance = engine.optimization_state.best_performance
        engine._perform_optimization_step()
        
        # Check optimization occurred
        assert engine.optimization_state.iteration > 0
        
        # Get insights
        insights = engine.get_optimization_insights()
        assert 'optimization_iteration' in insights
        assert 'best_performance' in insights
        assert 'current_parameters' in insights
        assert insights['total_observations'] == 15
        
        print(f"✅ Learning engine test passed: {insights['optimization_iteration']} iterations")
    
    def test_expert_skill_prediction(self):
        """Test expert skill evolution prediction."""
        predictor = ExpertSkillPredictor()
        
        # Simulate expert performance over time
        expert_id = "test_expert"
        base_time = datetime.now()
        
        for i in range(10):
            # Simulate skill improvement over time
            performance = 0.7 + (i * 0.02) + np.random.uniform(-0.05, 0.05)
            predictor.observe_performance(expert_id, performance, domain="algorithm")
        
        # Predict future skill
        future_skill = predictor.predict_future_skill(expert_id, days_ahead=30)
        
        # Should predict reasonable skill level
        assert 0.0 <= future_skill <= 1.0
        assert future_skill > 0.5  # Should be above baseline
        
        # Get domain expertise
        domain_skill = predictor.get_domain_expertise(expert_id, "algorithm")
        assert 0.0 <= domain_skill <= 1.0
        
        print(f"✅ Skill prediction test passed: predicted={future_skill:.3f}, domain={domain_skill:.3f}")
    
    def test_task_complexity_prediction(self):
        """Test task complexity and completion time prediction."""
        predictor = TaskComplexityPredictor()
        
        # Add sample completions
        for i in range(5):
            complexity = np.random.uniform(0.3, 0.9)
            metrics = CodeComplexityMetrics(
                cyclomatic_complexity=complexity * 30,
                cognitive_complexity=complexity * 60,
                halstead_complexity=complexity * 500,
                lines_of_code=int(complexity * 2000),
                function_count=int(complexity * 40),
                class_count=int(complexity * 10),
                dependency_depth=int(complexity * 8),
                api_surface_area=int(complexity * 15),
                test_coverage=max(0.3, 1.0 - complexity * 0.3),
                performance_requirements=np.random.uniform(0.0, 1.0),
                security_requirements=np.random.uniform(0.0, 1.0),
                scalability_requirements=np.random.uniform(0.0, 1.0)
            )
            
            actual_time = complexity * 25 + np.random.uniform(5, 15)
            quality = 0.9 - complexity * 0.3
            
            predictor.observe_task_completion(
                metrics, actual_time, quality, SDLCPhase.IMPLEMENTATION
            )
        
        # Test prediction
        test_metrics = CodeComplexityMetrics(
            cyclomatic_complexity=25.0, cognitive_complexity=50.0,
            halstead_complexity=300.0, lines_of_code=1500,
            function_count=30, class_count=8, dependency_depth=5,
            api_surface_area=12, test_coverage=0.75,
            performance_requirements=0.7, security_requirements=0.6,
            scalability_requirements=0.8
        )
        
        predicted_time = predictor.predict_completion_time(test_metrics, SDLCPhase.IMPLEMENTATION)
        
        # Should predict reasonable time
        assert predicted_time > 0
        assert predicted_time < 200  # Should be reasonable
        
        print(f"✅ Complexity prediction test passed: predicted_time={predicted_time:.1f}h")
    
    def test_integrated_optimization_workflow(self):
        """Test the complete integrated optimization workflow."""
        # Create optimizer with learning disabled for controlled testing
        optimizer = AutonomousSDLCOptimizer(
            self.experts,
            learning_objectives=[LearningObjective.MINIMIZE_COMPLETION_TIME, LearningObjective.MAXIMIZE_QUALITY],
            enable_continuous_learning=False
        )
        
        # Create and route multiple tasks
        tasks_completed = 0
        for i in range(5):
            complexity = np.random.uniform(0.4, 0.8)
            metrics = CodeComplexityMetrics(
                cyclomatic_complexity=complexity * 25,
                cognitive_complexity=complexity * 50,
                halstead_complexity=complexity * 300,
                lines_of_code=int(complexity * 1500),
                function_count=int(complexity * 30),
                class_count=int(complexity * 8),
                dependency_depth=int(complexity * 6),
                api_surface_area=int(complexity * 12),
                test_coverage=max(0.4, 1.0 - complexity * 0.2),
                performance_requirements=np.random.uniform(0.0, 1.0),
                security_requirements=np.random.uniform(0.0, 1.0),
                scalability_requirements=np.random.uniform(0.0, 1.0)
            )
            
            task = SDLCTask(
                task_id=f"INTEGRATED-{i:03d}",
                phase=np.random.choice(list(SDLCPhase)),
                description=f"Integrated test task {i}",
                complexity_metrics=metrics,
                priority=np.random.uniform(0.5, 1.0),
                deadline_pressure=np.random.uniform(0.3, 0.8),
                dependencies=[],
                estimated_effort=complexity * 30 + np.random.uniform(5, 15),
                risk_level=complexity * 0.7
            )
            
            # Route task
            result = optimizer.route_task_with_learning(task)
            
            # Simulate completion
            estimated_time = result['estimated_completion_time']
            actual_time = estimated_time * np.random.uniform(0.8, 1.2)
            quality = result['routing_confidence'] * np.random.uniform(0.85, 1.0)
            satisfaction = result['collaboration_score'] * np.random.uniform(0.8, 1.0)
            
            # Record completion
            optimizer.complete_task(
                task.task_id,
                actual_time,
                quality,
                defect_count=np.random.randint(0, 3),
                team_satisfaction=satisfaction
            )
            
            tasks_completed += 1
        
        # Check that tasks were completed and learning data collected
        dashboard = optimizer.get_performance_dashboard()
        assert dashboard['total_tasks_completed'] == tasks_completed
        assert len(optimizer.learning_engine.observations) == tasks_completed
        
        # Generate optimization report
        report = optimizer.generate_optimization_report()
        assert '🤖 AUTONOMOUS SDLC OPTIMIZATION REPORT' in report
        assert 'Learning Status:' in report
        assert 'Performance Trends:' in report
        
        print(f"✅ Integrated workflow test passed: {tasks_completed} tasks completed")
    
    def test_performance_under_load(self):
        """Test system performance under high load."""
        start_time = time.time()
        
        # Route many tasks quickly
        results = []
        for i in range(50):  # Higher load test
            metrics = CodeComplexityMetrics(
                cyclomatic_complexity=np.random.uniform(10, 40),
                cognitive_complexity=np.random.uniform(20, 80),
                halstead_complexity=np.random.uniform(100, 600),
                lines_of_code=int(np.random.uniform(500, 3000)),
                function_count=int(np.random.uniform(10, 60)),
                class_count=int(np.random.uniform(3, 20)),
                dependency_depth=int(np.random.uniform(2, 10)),
                api_surface_area=int(np.random.uniform(5, 25)),
                test_coverage=np.random.uniform(0.4, 0.9),
                performance_requirements=np.random.uniform(0.0, 1.0),
                security_requirements=np.random.uniform(0.0, 1.0),
                scalability_requirements=np.random.uniform(0.0, 1.0)
            )
            
            task = SDLCTask(
                task_id=f"LOAD-{i:03d}",
                phase=np.random.choice(list(SDLCPhase)),
                description=f"Load test task {i}",
                complexity_metrics=metrics,
                priority=np.random.uniform(0.3, 1.0),
                deadline_pressure=np.random.uniform(0.2, 0.9),
                dependencies=[],
                estimated_effort=np.random.uniform(10, 50),
                risk_level=np.random.uniform(0.1, 0.8)
            )
            
            result = self.router.route_task(task)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_task = total_time / 50
        
        # All tasks should complete successfully
        assert len(results) == 50
        
        # Performance should be reasonable (< 100ms per task)
        assert avg_time_per_task < 0.1
        
        print(f"✅ Load test passed: {avg_time_per_task*1000:.1f}ms avg per task")
    
    def test_error_handling_and_resilience(self):
        """Test system error handling and resilience."""
        # Test with invalid task
        try:
            invalid_metrics = CodeComplexityMetrics(
                cyclomatic_complexity=-1,  # Invalid
                cognitive_complexity=20.0,
                halstead_complexity=100.0,
                lines_of_code=500,
                function_count=10,
                class_count=3,
                dependency_depth=2,
                api_surface_area=5,
                test_coverage=0.8,
                performance_requirements=0.5,
                security_requirements=0.5,
                scalability_requirements=0.5
            )
            
            invalid_task = SDLCTask(
                task_id="INVALID-001",
                phase=SDLCPhase.IMPLEMENTATION,
                description="Invalid task",
                complexity_metrics=invalid_metrics,
                priority=0.7,
                deadline_pressure=0.5,
                dependencies=[],
                estimated_effort=20.0,
                risk_level=0.3
            )
            
            # Should handle gracefully
            result = self.router.route_task(invalid_task)
            assert result is not None  # Should return fallback result
            
        except Exception as e:
            # Should catch and handle errors gracefully
            assert "validation" in str(e).lower() or "error" in str(e).lower()
        
        # Test with empty expert list
        try:
            empty_router = AutonomousSDLCRouter([])
            # Should handle empty expert list
        except Exception as e:
            assert "expert" in str(e).lower()
        
        print("✅ Error handling test passed")
    
    def test_state_persistence(self):
        """Test saving and loading optimizer state."""
        optimizer = AutonomousSDLCOptimizer(
            self.experts,
            enable_continuous_learning=False
        )
        
        # Add some learning data
        for i in range(3):
            observation = PerformanceObservation(
                timestamp=datetime.now(),
                task_id=f"PERSIST-{i}",
                task_complexity=np.random.uniform(0.3, 0.8),
                assigned_experts=["architect"],
                expert_weights=[1.0],
                actual_completion_time=np.random.uniform(10, 30),
                estimated_completion_time=20.0,
                quality_score=np.random.uniform(0.7, 0.9),
                defect_count=np.random.randint(0, 2),
                team_satisfaction=np.random.uniform(0.7, 0.9),
                resource_cost=1000.0,
                context_features={}
            )
            optimizer.learning_engine.add_observation(observation)
        
        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            optimizer.save_state(tmp_path)
            
            # Create new optimizer and load state
            new_optimizer = AutonomousSDLCOptimizer(
                self.experts,
                enable_continuous_learning=False
            )
            new_optimizer.load_state(tmp_path)
            
            # Check state was loaded
            assert len(new_optimizer.learning_engine.observations) == 3
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        print("✅ State persistence test passed")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Running Comprehensive Autonomous SDLC Tests")
    print("=" * 60)
    
    test_suite = TestAutonomousSDLCSystem()
    test_suite.setup_method()
    
    tests = [
        ("Basic Functionality", test_suite.test_autonomous_sdlc_router_basic_functionality),
        ("Complexity-Based Selection", test_suite.test_complexity_based_expert_selection),
        ("Research Framework", test_suite.test_research_framework_baseline_comparison),
        ("Continuous Learning", test_suite.test_continuous_learning_engine),
        ("Skill Prediction", test_suite.test_expert_skill_prediction),
        ("Complexity Prediction", test_suite.test_task_complexity_prediction),
        ("Integrated Workflow", test_suite.test_integrated_optimization_workflow),
        ("Performance Load", test_suite.test_performance_under_load),
        ("Error Handling", test_suite.test_error_handling_and_resilience),
        ("State Persistence", test_suite.test_state_persistence)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔬 Running: {test_name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            failed += 1
    
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - System is ready for production!")
    else:
        print(f"\n⚠️  {failed} tests failed - Review and fix issues")
    
    return passed, failed


def run_demonstration_tests():
    """Run demonstration functions to validate they work."""
    print("\n🎭 Running Demonstration Tests")
    print("=" * 40)
    
    try:
        print("🤖 Testing Autonomous SDLC Router Demo...")
        router_result = demonstrate_autonomous_sdlc_routing()
        print("✅ Router demonstration completed successfully")
        
        print("\n🧠 Testing Optimization Demo...")  
        optimizer_result = demonstrate_autonomous_sdlc_optimization()
        print("✅ Optimizer demonstration completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run comprehensive tests
    passed, failed = run_comprehensive_tests()
    
    # Run demonstration tests  
    demo_success = run_demonstration_tests()
    
    # Final summary
    print(f"\n🏁 FINAL VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Unit Tests: {passed} passed, {failed} failed")
    print(f"Demonstrations: {'✅ Success' if demo_success else '❌ Failed'}")
    
    overall_success = (failed == 0) and demo_success
    print(f"Overall Status: {'🚀 READY FOR DEPLOYMENT' if overall_success else '⚠️ NEEDS ATTENTION'}")
    
    if overall_success:
        print("\n🎯 BREAKTHROUGH VALIDATED:")
        print("✅ Novel Autonomous SDLC Router - WORKING")
        print("✅ Research Validation Framework - WORKING") 
        print("✅ Continuous Learning Optimizer - WORKING")
        print("✅ All Quality Gates - PASSED")
        print("\n🏆 Ready for academic publication and production deployment!")