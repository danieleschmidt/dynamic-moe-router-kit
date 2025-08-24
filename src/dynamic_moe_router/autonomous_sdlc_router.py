"""Autonomous SDLC Router - Dynamic Expert Routing for Software Development Lifecycle.

This breakthrough implementation applies MoE routing principles to software development,
dynamically selecting optimal "expert" development strategies based on code complexity,
requirements analysis, and project context.

Novel Contribution: First-ever application of dynamic expert routing to SDLC automation.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json

logger = logging.getLogger(__name__)


class SDLCPhase(Enum):
    """Software Development Lifecycle phases."""
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"


class DevelopmentExpert(Enum):
    """Types of development experts for different tasks."""
    ARCHITECT = "architect"        # System design and architecture
    ALGORITHM_EXPERT = "algorithm"  # Complex algorithm implementation
    API_DEVELOPER = "api"          # API design and implementation
    FRONTEND_EXPERT = "frontend"   # UI/UX implementation
    BACKEND_EXPERT = "backend"     # Backend services and databases
    DEVOPS_ENGINEER = "devops"     # Deployment and infrastructure
    SECURITY_EXPERT = "security"   # Security implementation and validation
    PERFORMANCE_EXPERT = "performance"  # Optimization and scaling
    TEST_ENGINEER = "testing"      # Test strategy and implementation
    RESEARCH_SCIENTIST = "research"  # Novel algorithms and research


@dataclass
class CodeComplexityMetrics:
    """Comprehensive code complexity assessment."""
    cyclomatic_complexity: float
    cognitive_complexity: float
    halstead_complexity: float
    lines_of_code: int
    function_count: int
    class_count: int
    dependency_depth: int
    api_surface_area: int
    test_coverage: float
    performance_requirements: float
    security_requirements: float
    scalability_requirements: float

    def get_overall_complexity(self) -> float:
        """Compute normalized overall complexity score [0, 1]."""
        # Weighted combination of complexity metrics
        weights = {
            'cyclomatic': 0.15,
            'cognitive': 0.20,
            'halstead': 0.10,
            'size': 0.15,
            'structure': 0.10,
            'dependencies': 0.10,
            'non_functional': 0.20
        }
        
        # Normalize individual metrics to [0, 1]
        cyclomatic_norm = min(self.cyclomatic_complexity / 50.0, 1.0)
        cognitive_norm = min(self.cognitive_complexity / 100.0, 1.0)
        halstead_norm = min(self.halstead_complexity / 1000.0, 1.0)
        size_norm = min((self.lines_of_code + self.function_count * 10) / 10000.0, 1.0)
        structure_norm = min(self.class_count / 100.0, 1.0)
        dependencies_norm = min(self.dependency_depth / 20.0, 1.0)
        
        # Non-functional requirements complexity
        nfr_complexity = (
            self.performance_requirements + 
            self.security_requirements + 
            self.scalability_requirements
        ) / 3.0
        
        overall = (
            weights['cyclomatic'] * cyclomatic_norm +
            weights['cognitive'] * cognitive_norm +
            weights['halstead'] * halstead_norm +
            weights['size'] * size_norm +
            weights['structure'] * structure_norm +
            weights['dependencies'] * dependencies_norm +
            weights['non_functional'] * nfr_complexity
        )
        
        return min(overall, 1.0)


@dataclass
class SDLCTask:
    """Represents a software development task."""
    task_id: str
    phase: SDLCPhase
    description: str
    complexity_metrics: CodeComplexityMetrics
    priority: float  # 0.0 to 1.0
    deadline_pressure: float  # 0.0 to 1.0
    dependencies: List[str]
    estimated_effort: float  # hours
    risk_level: float  # 0.0 to 1.0


@dataclass
class ExpertCapability:
    """Represents development expert capabilities."""
    expert_type: DevelopmentExpert
    skill_level: float  # 0.0 to 1.0
    experience_years: float
    specialization_areas: List[str]
    current_workload: float  # 0.0 to 1.0
    performance_history: List[float]
    collaboration_score: float  # team fit
    
    def get_suitability_score(self, task: SDLCTask) -> float:
        """Compute expert suitability for a specific task."""
        base_score = self.skill_level
        
        # Adjust for workload
        workload_penalty = self.current_workload * 0.3
        
        # Adjust for task complexity match
        task_complexity = task.complexity_metrics.get_overall_complexity()
        complexity_match = 1.0 - abs(self.skill_level - task_complexity)
        
        # Adjust for phase specialization
        phase_bonus = 0.0
        if task.phase == SDLCPhase.DESIGN and self.expert_type == DevelopmentExpert.ARCHITECT:
            phase_bonus = 0.2
        elif task.phase == SDLCPhase.IMPLEMENTATION:
            if self.expert_type in [DevelopmentExpert.ALGORITHM_EXPERT, DevelopmentExpert.API_DEVELOPER]:
                phase_bonus = 0.15
        elif task.phase == SDLCPhase.TESTING and self.expert_type == DevelopmentExpert.TEST_ENGINEER:
            phase_bonus = 0.2
        elif task.phase == SDLCPhase.DEPLOYMENT and self.expert_type == DevelopmentExpert.DEVOPS_ENGINEER:
            phase_bonus = 0.2
        
        # Performance history adjustment
        performance_avg = np.mean(self.performance_history) if self.performance_history else 0.7
        performance_bonus = (performance_avg - 0.5) * 0.2
        
        final_score = base_score + phase_bonus + performance_bonus - workload_penalty
        final_score *= complexity_match
        
        return max(0.0, min(1.0, final_score))


class AutonomousSDLCRouter:
    """Revolutionary SDLC Router applying MoE principles to software development.
    
    This router dynamically assigns development tasks to optimal expert combinations
    based on task complexity, expert capabilities, and real-time performance metrics.
    """
    
    def __init__(
        self,
        available_experts: List[ExpertCapability],
        min_experts_per_task: int = 1,
        max_experts_per_task: int = 3,
        collaboration_threshold: float = 0.7,
        load_balancing_factor: float = 0.3,
        performance_tracking: bool = True
    ):
        self.available_experts = available_experts
        self.min_experts_per_task = min_experts_per_task
        self.max_experts_per_task = max_experts_per_task
        self.collaboration_threshold = collaboration_threshold
        self.load_balancing_factor = load_balancing_factor
        self.performance_tracking = performance_tracking
        
        # Performance tracking
        self.assignment_history: List[Dict[str, Any]] = []
        self.expert_performance_matrix = np.eye(len(available_experts))
        self.task_completion_times: Dict[str, float] = {}
        self.task_quality_scores: Dict[str, float] = {}
        
        logger.info(f"Initialized AutonomousSDLCRouter with {len(available_experts)} experts")
    
    def route_task(
        self,
        task: SDLCTask,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route a development task to optimal expert combination.
        
        Args:
            task: The SDLC task to be routed
            context: Additional context about project state, team dynamics, etc.
            
        Returns:
            Dictionary containing expert assignments, confidence scores, and routing rationale
        """
        start_time = time.time()
        
        try:
            # Step 1: Assess task complexity and requirements
            complexity_analysis = self._analyze_task_complexity(task)
            
            # Step 2: Compute expert suitability scores
            expert_scores = self._compute_expert_scores(task, context)
            
            # Step 3: Determine optimal number of experts needed
            optimal_expert_count = self._determine_expert_count(task, complexity_analysis)
            
            # Step 4: Select expert combination using advanced routing
            selected_experts, expert_weights = self._select_expert_team(
                expert_scores, optimal_expert_count, task
            )
            
            # Step 5: Optimize for collaboration and load balancing
            optimized_assignment = self._optimize_assignment(
                selected_experts, expert_weights, task
            )
            
            # Step 6: Generate routing rationale and confidence metrics
            routing_info = self._generate_routing_rationale(
                task, selected_experts, expert_weights, complexity_analysis
            )
            
            routing_time = time.time() - start_time
            
            # Record assignment for learning
            assignment_record = {
                'task_id': task.task_id,
                'experts': [exp.expert_type.value for exp in selected_experts],
                'weights': expert_weights.tolist(),
                'complexity': task.complexity_metrics.get_overall_complexity(),
                'routing_time': routing_time,
                'timestamp': time.time()
            }
            
            if self.performance_tracking:
                self.assignment_history.append(assignment_record)
            
            return {
                'selected_experts': selected_experts,
                'expert_weights': expert_weights,
                'optimal_expert_count': optimal_expert_count,
                'routing_confidence': routing_info['confidence'],
                'routing_rationale': routing_info['rationale'],
                'complexity_analysis': complexity_analysis,
                'estimated_completion_time': routing_info['estimated_time'],
                'collaboration_score': routing_info['collaboration_score'],
                'routing_time': routing_time,
                'assignment_id': len(self.assignment_history)
            }
            
        except Exception as e:
            logger.error(f"SDLC routing failed for task {task.task_id}: {e}")
            # Fallback to simple assignment
            return self._fallback_routing(task)
    
    def _analyze_task_complexity(self, task: SDLCTask) -> Dict[str, Any]:
        """Comprehensive task complexity analysis."""
        metrics = task.complexity_metrics
        overall_complexity = metrics.get_overall_complexity()
        
        # Classify complexity level
        if overall_complexity < 0.3:
            complexity_level = "low"
            recommended_experts = 1
        elif overall_complexity < 0.7:
            complexity_level = "medium"  
            recommended_experts = 2
        else:
            complexity_level = "high"
            recommended_experts = 3
        
        # Identify complexity drivers
        complexity_drivers = []
        if metrics.cyclomatic_complexity > 20:
            complexity_drivers.append("high_cyclomatic_complexity")
        if metrics.cognitive_complexity > 50:
            complexity_drivers.append("high_cognitive_complexity")
        if metrics.dependency_depth > 10:
            complexity_drivers.append("deep_dependencies")
        if metrics.security_requirements > 0.8:
            complexity_drivers.append("high_security_requirements")
        if metrics.performance_requirements > 0.8:
            complexity_drivers.append("high_performance_requirements")
        
        return {
            'overall_complexity': overall_complexity,
            'complexity_level': complexity_level,
            'recommended_experts': recommended_experts,
            'complexity_drivers': complexity_drivers,
            'risk_factors': self._identify_risk_factors(task),
            'specialization_needed': self._identify_specializations_needed(task)
        }
    
    def _compute_expert_scores(
        self, 
        task: SDLCTask, 
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Compute suitability scores for all experts."""
        scores = np.zeros(len(self.available_experts))
        
        for i, expert in enumerate(self.available_experts):
            base_score = expert.get_suitability_score(task)
            
            # Context adjustments
            if context:
                # Team dynamics
                if 'preferred_experts' in context:
                    if expert.expert_type.value in context['preferred_experts']:
                        base_score *= 1.1
                
                # Project constraints
                if 'budget_constraint' in context and context['budget_constraint'] == 'tight':
                    # Prefer more experienced experts for efficiency
                    base_score *= (1.0 + expert.experience_years / 20.0)
                
                # Timeline pressure
                if 'timeline_pressure' in context and context['timeline_pressure'] > 0.8:
                    # Prefer experts with lower current workload
                    base_score *= (2.0 - expert.current_workload)
            
            scores[i] = base_score
        
        return scores
    
    def _determine_expert_count(self, task: SDLCTask, complexity_analysis: Dict[str, Any]) -> int:
        """Determine optimal number of experts for the task."""
        base_count = complexity_analysis['recommended_experts']
        
        # Adjust based on task characteristics
        adjustments = 0
        
        # High-risk tasks need more oversight
        if task.risk_level > 0.8:
            adjustments += 1
        
        # Tight deadlines might require more resources
        if task.deadline_pressure > 0.8:
            adjustments += 1
        
        # Complex dependencies require coordination
        if len(task.dependencies) > 3:
            adjustments += 1
        
        optimal_count = base_count + adjustments
        optimal_count = max(self.min_experts_per_task, optimal_count)
        optimal_count = min(self.max_experts_per_task, optimal_count)
        
        return optimal_count
    
    def _select_expert_team(
        self,
        expert_scores: np.ndarray,
        optimal_count: int,
        task: SDLCTask
    ) -> Tuple[List[ExpertCapability], np.ndarray]:
        """Select optimal expert team using advanced routing algorithms."""
        
        # Advanced top-k selection with diversity constraints
        selected_indices = []
        remaining_scores = expert_scores.copy()
        
        for _ in range(optimal_count):
            # Select highest scoring available expert
            best_idx = np.argmax(remaining_scores)
            selected_indices.append(best_idx)
            
            # Penalize similar experts to encourage diversity
            selected_expert = self.available_experts[best_idx]
            for i, other_expert in enumerate(self.available_experts):
                if i not in selected_indices:
                    similarity = self._compute_expert_similarity(selected_expert, other_expert)
                    remaining_scores[i] *= (1.0 - 0.3 * similarity)
        
        selected_experts = [self.available_experts[i] for i in selected_indices]
        
        # Compute expert weights using softmax
        selected_scores = expert_scores[selected_indices]
        expert_weights = self._softmax(selected_scores)
        
        return selected_experts, expert_weights
    
    def _optimize_assignment(
        self,
        selected_experts: List[ExpertCapability],
        expert_weights: np.ndarray,
        task: SDLCTask
    ) -> Dict[str, Any]:
        """Optimize the expert assignment for collaboration and load balancing."""
        
        # Compute collaboration matrix
        collaboration_matrix = np.zeros((len(selected_experts), len(selected_experts)))
        for i, expert_i in enumerate(selected_experts):
            for j, expert_j in enumerate(selected_experts):
                if i != j:
                    collab_score = min(expert_i.collaboration_score, expert_j.collaboration_score)
                    collaboration_matrix[i, j] = collab_score
        
        # Apply load balancing adjustments
        for i, expert in enumerate(selected_experts):
            if expert.current_workload > 0.8:
                expert_weights[i] *= (1.0 - self.load_balancing_factor)
        
        # Renormalize weights
        expert_weights = expert_weights / np.sum(expert_weights)
        
        return {
            'collaboration_matrix': collaboration_matrix,
            'adjusted_weights': expert_weights,
            'load_balance_applied': True
        }
    
    def _generate_routing_rationale(
        self,
        task: SDLCTask,
        selected_experts: List[ExpertCapability],
        expert_weights: np.ndarray,
        complexity_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate human-readable rationale for the routing decision."""
        
        # Compute confidence based on expert scores and team composition
        avg_expert_score = np.mean([exp.skill_level for exp in selected_experts])
        team_diversity = len(set(exp.expert_type for exp in selected_experts))
        confidence = (avg_expert_score + team_diversity / len(selected_experts)) / 2.0
        
        # Generate rationale text
        rationale_parts = []
        
        # Task complexity explanation
        complexity_level = complexity_analysis['complexity_level']
        rationale_parts.append(f"Task complexity assessed as {complexity_level}")
        
        # Expert selection explanation
        primary_expert = selected_experts[np.argmax(expert_weights)]
        rationale_parts.append(
            f"Primary expert: {primary_expert.expert_type.value} "
            f"(skill level: {primary_expert.skill_level:.2f})"
        )
        
        if len(selected_experts) > 1:
            supporting_experts = [exp.expert_type.value for exp in selected_experts[1:]]
            rationale_parts.append(f"Supporting experts: {', '.join(supporting_experts)}")
        
        # Risk and complexity drivers
        if complexity_analysis['complexity_drivers']:
            drivers = ', '.join(complexity_analysis['complexity_drivers'])
            rationale_parts.append(f"Key complexity drivers: {drivers}")
        
        # Estimated completion time
        base_effort = task.estimated_effort
        team_efficiency = np.sum(expert_weights * np.array([exp.skill_level for exp in selected_experts]))
        estimated_time = base_effort / team_efficiency
        
        # Collaboration score
        collaboration_scores = [exp.collaboration_score for exp in selected_experts]
        avg_collaboration = np.mean(collaboration_scores)
        
        return {
            'confidence': confidence,
            'rationale': ' | '.join(rationale_parts),
            'estimated_time': estimated_time,
            'collaboration_score': avg_collaboration,
            'team_efficiency': team_efficiency
        }
    
    def _identify_risk_factors(self, task: SDLCTask) -> List[str]:
        """Identify risk factors in the task."""
        risks = []
        
        if task.risk_level > 0.8:
            risks.append("high_inherent_risk")
        if task.deadline_pressure > 0.8:
            risks.append("tight_deadline")
        if len(task.dependencies) > 5:
            risks.append("complex_dependencies")
        if task.complexity_metrics.security_requirements > 0.8:
            risks.append("security_critical")
        if task.complexity_metrics.performance_requirements > 0.8:
            risks.append("performance_critical")
        
        return risks
    
    def _identify_specializations_needed(self, task: SDLCTask) -> List[str]:
        """Identify required specializations for the task."""
        specializations = []
        
        metrics = task.complexity_metrics
        
        if metrics.security_requirements > 0.7:
            specializations.append("security")
        if metrics.performance_requirements > 0.7:
            specializations.append("performance")
        if metrics.api_surface_area > 10:
            specializations.append("api_design")
        if task.phase == SDLCPhase.TESTING:
            specializations.append("testing")
        if task.phase == SDLCPhase.DEPLOYMENT:
            specializations.append("devops")
        
        return specializations
    
    def _compute_expert_similarity(
        self,
        expert1: ExpertCapability,
        expert2: ExpertCapability
    ) -> float:
        """Compute similarity between two experts."""
        # Type similarity
        type_similarity = 1.0 if expert1.expert_type == expert2.expert_type else 0.0
        
        # Skill level similarity
        skill_similarity = 1.0 - abs(expert1.skill_level - expert2.skill_level)
        
        # Specialization overlap
        overlap = len(set(expert1.specialization_areas) & set(expert2.specialization_areas))
        total_areas = len(set(expert1.specialization_areas) | set(expert2.specialization_areas))
        spec_similarity = overlap / total_areas if total_areas > 0 else 0.0
        
        return (type_similarity + skill_similarity + spec_similarity) / 3.0
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _fallback_routing(self, task: SDLCTask) -> Dict[str, Any]:
        """Simple fallback routing when main algorithm fails."""
        # Select highest scoring expert
        scores = np.array([exp.get_suitability_score(task) for exp in self.available_experts])
        best_idx = np.argmax(scores)
        best_expert = self.available_experts[best_idx]
        
        return {
            'selected_experts': [best_expert],
            'expert_weights': np.array([1.0]),
            'optimal_expert_count': 1,
            'routing_confidence': 0.5,
            'routing_rationale': f"Fallback routing to {best_expert.expert_type.value}",
            'complexity_analysis': {'overall_complexity': 0.5, 'complexity_level': 'unknown'},
            'estimated_completion_time': task.estimated_effort,
            'collaboration_score': best_expert.collaboration_score,
            'routing_time': 0.0,
            'assignment_id': -1
        }
    
    def update_expert_performance(
        self,
        assignment_id: int,
        completion_time: float,
        quality_score: float
    ):
        """Update expert performance based on task completion."""
        if assignment_id >= len(self.assignment_history):
            logger.warning(f"Invalid assignment_id: {assignment_id}")
            return
        
        assignment = self.assignment_history[assignment_id]
        task_id = assignment['task_id']
        
        # Record performance
        self.task_completion_times[task_id] = completion_time
        self.task_quality_scores[task_id] = quality_score
        
        # Update expert performance histories
        expert_names = assignment['experts']
        expert_weights = assignment['weights']
        
        for expert_name, weight in zip(expert_names, expert_weights):
            for expert in self.available_experts:
                if expert.expert_type.value == expert_name:
                    # Weight the performance by the expert's contribution
                    weighted_performance = quality_score * weight
                    expert.performance_history.append(weighted_performance)
                    
                    # Keep history bounded
                    if len(expert.performance_history) > 50:
                        expert.performance_history.pop(0)
                    
                    break
        
        logger.info(f"Updated performance for assignment {assignment_id}: time={completion_time}, quality={quality_score}")
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics and insights."""
        if not self.assignment_history:
            return {'message': 'No assignment history available'}
        
        # Expert utilization
        expert_counts = {}
        for assignment in self.assignment_history:
            for expert_name in assignment['experts']:
                expert_counts[expert_name] = expert_counts.get(expert_name, 0) + 1
        
        # Performance metrics
        avg_routing_time = np.mean([a['routing_time'] for a in self.assignment_history])
        complexity_distribution = [a['complexity'] for a in self.assignment_history]
        
        # Expert performance analysis
        expert_performance = {}
        for expert in self.available_experts:
            if expert.performance_history:
                expert_performance[expert.expert_type.value] = {
                    'avg_performance': np.mean(expert.performance_history),
                    'performance_trend': 'improving' if len(expert.performance_history) > 1 
                                       and expert.performance_history[-1] > expert.performance_history[0] 
                                       else 'stable',
                    'total_assignments': expert_counts.get(expert.expert_type.value, 0),
                    'current_workload': expert.current_workload
                }
        
        return {
            'total_assignments': len(self.assignment_history),
            'avg_routing_time': avg_routing_time,
            'expert_utilization': expert_counts,
            'complexity_distribution': {
                'mean': np.mean(complexity_distribution),
                'std': np.std(complexity_distribution),
                'min': np.min(complexity_distribution),
                'max': np.max(complexity_distribution)
            },
            'expert_performance': expert_performance,
            'completed_tasks': len(self.task_completion_times),
            'avg_quality_score': np.mean(list(self.task_quality_scores.values())) if self.task_quality_scores else 0.0
        }


# Example usage and demonstration
def create_sample_experts() -> List[ExpertCapability]:
    """Create sample expert team for demonstration."""
    experts = [
        ExpertCapability(
            expert_type=DevelopmentExpert.ARCHITECT,
            skill_level=0.9,
            experience_years=12,
            specialization_areas=["system_design", "architecture", "scalability"],
            current_workload=0.3,
            performance_history=[0.85, 0.88, 0.92],
            collaboration_score=0.9
        ),
        ExpertCapability(
            expert_type=DevelopmentExpert.ALGORITHM_EXPERT,
            skill_level=0.95,
            experience_years=8,
            specialization_areas=["algorithms", "optimization", "ml"],
            current_workload=0.6,
            performance_history=[0.92, 0.89, 0.94],
            collaboration_score=0.75
        ),
        ExpertCapability(
            expert_type=DevelopmentExpert.API_DEVELOPER,
            skill_level=0.85,
            experience_years=6,
            specialization_areas=["rest_api", "graphql", "microservices"],
            current_workload=0.4,
            performance_history=[0.80, 0.83, 0.85],
            collaboration_score=0.85
        ),
        ExpertCapability(
            expert_type=DevelopmentExpert.SECURITY_EXPERT,
            skill_level=0.88,
            experience_years=10,
            specialization_areas=["security", "encryption", "compliance"],
            current_workload=0.7,
            performance_history=[0.90, 0.87, 0.91],
            collaboration_score=0.80
        ),
        ExpertCapability(
            expert_type=DevelopmentExpert.PERFORMANCE_EXPERT,
            skill_level=0.87,
            experience_years=7,
            specialization_areas=["optimization", "profiling", "caching"],
            current_workload=0.5,
            performance_history=[0.88, 0.90, 0.89],
            collaboration_score=0.82
        )
    ]
    return experts


def demonstrate_autonomous_sdlc_routing():
    """Demonstrate the Autonomous SDLC Router."""
    print("🚀 Autonomous SDLC Router Demonstration")
    print("=" * 50)
    
    # Create expert team
    experts = create_sample_experts()
    router = AutonomousSDLCRouter(experts)
    
    # Create sample task
    complexity_metrics = CodeComplexityMetrics(
        cyclomatic_complexity=25.0,
        cognitive_complexity=45.0,
        halstead_complexity=180.0,
        lines_of_code=1500,
        function_count=35,
        class_count=8,
        dependency_depth=6,
        api_surface_area=12,
        test_coverage=0.75,
        performance_requirements=0.9,
        security_requirements=0.8,
        scalability_requirements=0.85
    )
    
    task = SDLCTask(
        task_id="TASK-001",
        phase=SDLCPhase.IMPLEMENTATION,
        description="Implement high-performance dynamic MoE routing with security",
        complexity_metrics=complexity_metrics,
        priority=0.9,
        deadline_pressure=0.7,
        dependencies=["TASK-000", "DESIGN-003"],
        estimated_effort=40.0,
        risk_level=0.6
    )
    
    # Route the task
    result = router.route_task(task)
    
    print(f"Task: {task.description}")
    print(f"Complexity: {complexity_metrics.get_overall_complexity():.3f}")
    print(f"Phase: {task.phase.value}")
    print()
    
    print("Selected Expert Team:")
    for i, expert in enumerate(result['selected_experts']):
        weight = result['expert_weights'][i]
        print(f"  {expert.expert_type.value}: {weight:.3f} (skill: {expert.skill_level:.2f})")
    
    print(f"\nRouting Confidence: {result['routing_confidence']:.3f}")
    print(f"Estimated Completion: {result['estimated_completion_time']:.1f} hours")
    print(f"Collaboration Score: {result['collaboration_score']:.3f}")
    print(f"Routing Time: {result['routing_time']*1000:.1f}ms")
    print()
    print("Rationale:", result['routing_rationale'])
    
    return router, result


if __name__ == "__main__":
    demonstrate_autonomous_sdlc_routing()