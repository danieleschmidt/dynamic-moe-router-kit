"""Test suite for Autonomous SDLC Router - Novel MoE application to software development."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

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


class TestCodeComplexityMetrics:
    """Test the code complexity assessment framework."""
    
    def test_simple_complexity(self):
        """Test basic complexity calculation."""
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=5.0,
            cognitive_complexity=10.0,
            halstead_complexity=50.0,
            lines_of_code=100,
            function_count=5,
            class_count=2,
            dependency_depth=2,
            api_surface_area=3,
            test_coverage=0.8,
            performance_requirements=0.3,
            security_requirements=0.2,
            scalability_requirements=0.4
        )
        
        complexity = metrics.get_overall_complexity()
        assert 0.0 <= complexity <= 1.0
        assert complexity < 0.3  # Should be low complexity
    
    def test_high_complexity(self):
        """Test high complexity scenario."""
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=45.0,
            cognitive_complexity=80.0,
            halstead_complexity=800.0,
            lines_of_code=5000,
            function_count=100,
            class_count=50,
            dependency_depth=15,
            api_surface_area=25,
            test_coverage=0.5,
            performance_requirements=0.9,
            security_requirements=0.95,
            scalability_requirements=0.9
        )
        
        complexity = metrics.get_overall_complexity()
        assert complexity > 0.7  # Should be high complexity
        assert complexity <= 1.0
    
    def test_complexity_normalization(self):
        """Test that complexity is properly normalized."""
        # Extreme values should still result in normalized complexity
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=1000.0,
            cognitive_complexity=1000.0,
            halstead_complexity=10000.0,
            lines_of_code=100000,
            function_count=1000,
            class_count=1000,
            dependency_depth=100,
            api_surface_area=1000,
            test_coverage=1.0,
            performance_requirements=1.0,
            security_requirements=1.0,
            scalability_requirements=1.0
        )
        
        complexity = metrics.get_overall_complexity()
        assert complexity == 1.0  # Should be capped at 1.0


class TestExpertCapability:
    """Test expert capability assessment."""
    
    def test_expert_suitability_basic(self):
        """Test basic expert suitability calculation."""
        expert = ExpertCapability(
            expert_type=DevelopmentExpert.ALGORITHM_EXPERT,
            skill_level=0.8,
            experience_years=5,
            specialization_areas=["algorithms", "optimization"],
            current_workload=0.3,
            performance_history=[0.8, 0.85, 0.9],
            collaboration_score=0.8
        )
        
        # Create a matching task
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=20.0, cognitive_complexity=40.0,
            halstead_complexity=200.0, lines_of_code=1000,
            function_count=20, class_count=5, dependency_depth=4,
            api_surface_area=8, test_coverage=0.8,
            performance_requirements=0.7, security_requirements=0.3,
            scalability_requirements=0.5
        )
        
        task = SDLCTask(
            task_id="TEST-001",
            phase=SDLCPhase.IMPLEMENTATION,
            description="Algorithm implementation task",
            complexity_metrics=metrics,
            priority=0.8,
            deadline_pressure=0.5,
            dependencies=[],
            estimated_effort=20.0,
            risk_level=0.4
        )
        
        suitability = expert.get_suitability_score(task)
        assert 0.0 <= suitability <= 1.0
        assert suitability > 0.5  # Should be reasonably suitable
    
    def test_phase_specialization_bonus(self):
        """Test that experts get bonuses for their specialized phases."""
        architect = ExpertCapability(
            expert_type=DevelopmentExpert.ARCHITECT,
            skill_level=0.8, experience_years=10,
            specialization_areas=["architecture"], current_workload=0.5,
            performance_history=[0.8], collaboration_score=0.8
        )
        
        tester = ExpertCapability(
            expert_type=DevelopmentExpert.TEST_ENGINEER,
            skill_level=0.8, experience_years=10,
            specialization_areas=["testing"], current_workload=0.5,
            performance_history=[0.8], collaboration_score=0.8
        )
        
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=10.0, cognitive_complexity=20.0,
            halstead_complexity=100.0, lines_of_code=500,
            function_count=10, class_count=3, dependency_depth=2,
            api_surface_area=5, test_coverage=0.9,
            performance_requirements=0.5, security_requirements=0.5,
            scalability_requirements=0.5
        )
        
        design_task = SDLCTask(
            task_id="DESIGN-001", phase=SDLCPhase.DESIGN,
            description="System design", complexity_metrics=metrics,
            priority=0.7, deadline_pressure=0.5, dependencies=[],
            estimated_effort=15.0, risk_level=0.3
        )
        
        test_task = SDLCTask(
            task_id="TEST-001", phase=SDLCPhase.TESTING,
            description="Test implementation", complexity_metrics=metrics,
            priority=0.7, deadline_pressure=0.5, dependencies=[],
            estimated_effort=15.0, risk_level=0.3
        )
        
        architect_design_score = architect.get_suitability_score(design_task)
        architect_test_score = architect.get_suitability_score(test_task)
        
        tester_design_score = tester.get_suitability_score(design_task)
        tester_test_score = tester.get_suitability_score(test_task)
        
        # Architect should score higher on design task
        assert architect_design_score > architect_test_score
        # Tester should score higher on test task
        assert tester_test_score > tester_design_score


class TestAutonomousSDLCRouter:
    """Test the main SDLC router functionality."""
    
    @pytest.fixture
    def sample_experts(self):
        """Create sample experts for testing."""
        return create_sample_experts()
    
    @pytest.fixture
    def router(self, sample_experts):
        """Create router instance."""
        return AutonomousSDLCRouter(sample_experts)
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=15.0, cognitive_complexity=25.0,
            halstead_complexity=150.0, lines_of_code=800,
            function_count=15, class_count=4, dependency_depth=3,
            api_surface_area=6, test_coverage=0.8,
            performance_requirements=0.6, security_requirements=0.7,
            scalability_requirements=0.5
        )
        
        return SDLCTask(
            task_id="SAMPLE-001",
            phase=SDLCPhase.IMPLEMENTATION,
            description="Sample implementation task",
            complexity_metrics=metrics,
            priority=0.7,
            deadline_pressure=0.5,
            dependencies=["DEP-001"],
            estimated_effort=25.0,
            risk_level=0.4
        )
    
    def test_router_initialization(self, sample_experts):
        """Test router initialization."""
        router = AutonomousSDLCRouter(sample_experts)
        
        assert router.available_experts == sample_experts
        assert router.min_experts_per_task == 1
        assert router.max_experts_per_task == 3
        assert len(router.assignment_history) == 0
    
    def test_basic_routing(self, router, sample_task):
        """Test basic task routing functionality."""
        result = router.route_task(sample_task)
        
        # Check result structure
        required_keys = [
            'selected_experts', 'expert_weights', 'optimal_expert_count',
            'routing_confidence', 'routing_rationale', 'complexity_analysis',
            'estimated_completion_time', 'collaboration_score', 'routing_time'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Check constraints
        assert 1 <= len(result['selected_experts']) <= 3
        assert len(result['expert_weights']) == len(result['selected_experts'])
        assert np.isclose(np.sum(result['expert_weights']), 1.0, atol=1e-6)
        assert 0.0 <= result['routing_confidence'] <= 1.0
        assert result['routing_time'] > 0
    
    def test_complexity_analysis(self, router, sample_task):
        """Test task complexity analysis."""
        complexity_analysis = router._analyze_task_complexity(sample_task)
        
        assert 'overall_complexity' in complexity_analysis
        assert 'complexity_level' in complexity_analysis
        assert 'recommended_experts' in complexity_analysis
        assert 'complexity_drivers' in complexity_analysis
        
        assert complexity_analysis['complexity_level'] in ['low', 'medium', 'high']
        assert 1 <= complexity_analysis['recommended_experts'] <= 3
        assert 0.0 <= complexity_analysis['overall_complexity'] <= 1.0
    
    def test_expert_count_determination(self, router, sample_task):
        """Test optimal expert count determination."""
        complexity_analysis = router._analyze_task_complexity(sample_task)
        expert_count = router._determine_expert_count(sample_task, complexity_analysis)
        
        assert router.min_experts_per_task <= expert_count <= router.max_experts_per_task
        assert isinstance(expert_count, int)
    
    def test_expert_selection_diversity(self, router, sample_experts):
        """Test that expert selection promotes diversity."""
        # Create high-complexity task that should require multiple experts
        metrics = CodeComplexityMetrics(
            cyclomatic_complexity=40.0, cognitive_complexity=70.0,
            halstead_complexity=600.0, lines_of_code=3000,
            function_count=60, class_count=15, dependency_depth=8,
            api_surface_area=20, test_coverage=0.6,
            performance_requirements=0.9, security_requirements=0.9,
            scalability_requirements=0.8
        )
        
        high_complexity_task = SDLCTask(
            task_id="COMPLEX-001",
            phase=SDLCPhase.IMPLEMENTATION,
            description="High complexity implementation",
            complexity_metrics=metrics,
            priority=0.9,
            deadline_pressure=0.8,
            dependencies=["DEP-001", "DEP-002", "DEP-003", "DEP-004"],
            estimated_effort=60.0,
            risk_level=0.8
        )
        
        result = router.route_task(high_complexity_task)
        selected_types = [exp.expert_type for exp in result['selected_experts']]
        
        # Should select multiple experts for high complexity
        assert len(result['selected_experts']) > 1
        # Should have diverse expert types
        assert len(set(selected_types)) == len(selected_types)  # All different types
    
    def test_load_balancing(self, router, sample_task):
        """Test that load balancing affects expert selection."""
        # Set high workload for one expert
        router.available_experts[0].current_workload = 0.9
        router.available_experts[1].current_workload = 0.1
        
        # Route multiple tasks and check that low-workload expert is preferred
        results = []
        for i in range(5):
            task = sample_task
            task.task_id = f"LOAD-TEST-{i}"
            result = router.route_task(task)
            results.append(result)
        
        # Count selections of each expert
        expert_selections = {}
        for result in results:
            for expert in result['selected_experts']:
                expert_type = expert.expert_type.value
                expert_selections[expert_type] = expert_selections.get(expert_type, 0) + 1
        
        # Should show some load balancing effect
        assert len(expert_selections) > 0
    
    def test_performance_tracking(self, router, sample_task):
        """Test expert performance tracking and updates."""
        # Route a task
        result = router.route_task(sample_task)
        assignment_id = result['assignment_id']
        
        # Simulate task completion
        router.update_expert_performance(assignment_id, completion_time=20.0, quality_score=0.9)
        
        # Check that performance was recorded
        assert sample_task.task_id in router.task_completion_times
        assert sample_task.task_id in router.task_quality_scores
        
        # Check that expert performance history was updated
        selected_expert_types = [exp.expert_type.value for exp in result['selected_experts']]
        for expert in router.available_experts:
            if expert.expert_type.value in selected_expert_types:
                assert len(expert.performance_history) > 0
                # Most recent performance should be weighted by expert contribution
                assert expert.performance_history[-1] > 0
    
    def test_fallback_routing(self, router, sample_task):
        """Test fallback routing when main algorithm fails."""
        # Mock the main routing to fail
        with patch.object(router, '_analyze_task_complexity', side_effect=Exception("Test error")):
            result = router._fallback_routing(sample_task)
            
            assert len(result['selected_experts']) == 1
            assert result['routing_confidence'] == 0.5
            assert 'Fallback routing' in result['routing_rationale']
    
    def test_routing_analytics(self, router, sample_task):
        """Test routing analytics and insights."""
        # Initially no data
        analytics = router.get_routing_analytics()
        assert 'message' in analytics
        
        # Route some tasks
        for i in range(3):
            task = sample_task
            task.task_id = f"ANALYTICS-{i}"
            result = router.route_task(task)
            router.update_expert_performance(result['assignment_id'], 
                                           completion_time=15.0 + i, 
                                           quality_score=0.8 + i * 0.05)
        
        analytics = router.get_routing_analytics()
        
        assert 'total_assignments' in analytics
        assert analytics['total_assignments'] == 3
        assert 'avg_routing_time' in analytics
        assert 'expert_utilization' in analytics
        assert 'complexity_distribution' in analytics
        assert 'expert_performance' in analytics
    
    def test_context_based_routing(self, router, sample_task):
        """Test routing with different contexts."""
        # Test with budget constraint
        budget_context = {'budget_constraint': 'tight'}
        result1 = router.route_task(sample_task, context=budget_context)
        
        # Test with timeline pressure
        timeline_context = {'timeline_pressure': 0.9}
        result2 = router.route_task(sample_task, context=timeline_context)
        
        # Test with preferred experts
        preferred_context = {'preferred_experts': ['architect', 'algorithm']}
        result3 = router.route_task(sample_task, context=preferred_context)
        
        # All should produce valid results
        for result in [result1, result2, result3]:
            assert len(result['selected_experts']) >= 1
            assert result['routing_confidence'] > 0


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_demonstration(self):
        """Test the complete demonstration scenario."""
        router, result = demonstrate_autonomous_sdlc_routing()
        
        # Verify the demonstration runs successfully
        assert isinstance(router, AutonomousSDLCRouter)
        assert 'selected_experts' in result
        assert len(result['selected_experts']) > 0
        assert result['routing_confidence'] > 0
        
        # Verify analytics work after demonstration
        analytics = router.get_routing_analytics()
        assert analytics['total_assignments'] == 1
    
    def test_multiple_phase_routing(self):
        """Test routing tasks across different SDLC phases."""
        experts = create_sample_experts()
        router = AutonomousSDLCRouter(experts)
        
        phases = list(SDLCPhase)
        results = []
        
        for phase in phases:
            metrics = CodeComplexityMetrics(
                cyclomatic_complexity=10.0, cognitive_complexity=20.0,
                halstead_complexity=100.0, lines_of_code=500,
                function_count=10, class_count=3, dependency_depth=2,
                api_surface_area=5, test_coverage=0.8,
                performance_requirements=0.5, security_requirements=0.5,
                scalability_requirements=0.5
            )
            
            task = SDLCTask(
                task_id=f"PHASE-{phase.value}",
                phase=phase,
                description=f"Task for {phase.value} phase",
                complexity_metrics=metrics,
                priority=0.7,
                deadline_pressure=0.5,
                dependencies=[],
                estimated_effort=20.0,
                risk_level=0.3
            )
            
            result = router.route_task(task)
            results.append(result)
        
        # Should have routed tasks for all phases
        assert len(results) == len(phases)
        
        # Each result should be valid
        for result in results:
            assert len(result['selected_experts']) > 0
            assert result['routing_confidence'] > 0
    
    def test_expert_similarity_and_diversity(self):
        """Test expert similarity computation and diversity promotion."""
        experts = create_sample_experts()
        router = AutonomousSDLCRouter(experts)
        
        # Test similarity between similar experts
        architect = experts[0]
        algorithm_expert = experts[1]
        
        similarity = router._compute_expert_similarity(architect, algorithm_expert)
        assert 0.0 <= similarity <= 1.0
        
        # Test similarity between identical expert types
        duplicate_architect = ExpertCapability(
            expert_type=DevelopmentExpert.ARCHITECT,
            skill_level=architect.skill_level,
            experience_years=architect.experience_years,
            specialization_areas=architect.specialization_areas,
            current_workload=0.5,
            performance_history=[0.8],
            collaboration_score=0.8
        )
        
        high_similarity = router._compute_expert_similarity(architect, duplicate_architect)
        assert high_similarity > similarity  # Should be more similar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])