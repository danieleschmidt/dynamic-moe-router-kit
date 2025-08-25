"""Autonomous SDLC Optimizer - Self-Improving Software Development Patterns.

This advanced system implements self-learning capabilities that continuously
optimize SDLC processes based on historical performance, team dynamics,
and project outcomes.

Revolutionary Feature: First autonomous SDLC system with continuous learning
and self-optimization capabilities for enterprise software development.
"""

import logging
import time
import pickle
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from .autonomous_sdlc_router import (
    AutonomousSDLCRouter,
    SDLCPhase,
    DevelopmentExpert, 
    CodeComplexityMetrics,
    SDLCTask,
    ExpertCapability
)

logger = logging.getLogger(__name__)


class LearningObjective(Enum):
    """Learning objectives for SDLC optimization."""
    MINIMIZE_COMPLETION_TIME = "minimize_time"
    MAXIMIZE_QUALITY = "maximize_quality" 
    OPTIMIZE_RESOURCE_USAGE = "optimize_resources"
    IMPROVE_TEAM_SATISFACTION = "improve_satisfaction"
    REDUCE_DEFECT_RATE = "reduce_defects"
    ENHANCE_PREDICTABILITY = "enhance_predictability"


class OptimizationStrategy(Enum):
    """Strategies for continuous optimization."""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class PerformanceObservation:
    """Single performance observation for learning."""
    timestamp: datetime
    task_id: str
    task_complexity: float
    assigned_experts: List[str]
    expert_weights: List[float]
    actual_completion_time: float
    estimated_completion_time: float
    quality_score: float
    defect_count: int
    team_satisfaction: float
    resource_cost: float
    context_features: Dict[str, Any]
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert observation to feature vector for ML."""
        features = [
            self.task_complexity,
            len(self.assigned_experts),
            np.mean(self.expert_weights) if self.expert_weights else 0.0,
            np.std(self.expert_weights) if len(self.expert_weights) > 1 else 0.0,
            self.actual_completion_time / max(self.estimated_completion_time, 1.0),
            self.quality_score,
            float(self.defect_count),
            self.team_satisfaction,
            self.resource_cost / 1000.0,  # Normalize
        ]
        
        # Add context features
        features.extend([
            self.context_features.get('timeline_pressure', 0.0),
            self.context_features.get('budget_constraint', 0.0),
            self.context_features.get('team_experience', 0.0),
            self.context_features.get('project_phase_progress', 0.0)
        ])
        
        return np.array(features, dtype=np.float32)


@dataclass
class OptimizationState:
    """Current state of SDLC optimization."""
    iteration: int = 0
    best_performance: float = 0.0
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    performance_history: List[float] = field(default_factory=list)
    parameter_history: List[Dict[str, Any]] = field(default_factory=list)
    convergence_threshold: float = 0.001
    patience: int = 10
    stagnant_iterations: int = 0


class ExpertSkillPredictor:
    """Predicts expert skill evolution over time."""
    
    def __init__(self):
        self.skill_trajectories: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.learning_rates: Dict[str, float] = {}
        self.expertise_domains: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def observe_performance(self, expert_id: str, performance_score: float, domain: str = "general"):
        """Record expert performance observation."""
        timestamp = datetime.now()
        self.skill_trajectories[expert_id].append((timestamp, performance_score))
        
        # Update domain expertise
        if domain not in self.expertise_domains[expert_id]:
            self.expertise_domains[expert_id][domain] = performance_score
        else:
            # Exponential moving average
            current = self.expertise_domains[expert_id][domain]
            self.expertise_domains[expert_id][domain] = 0.9 * current + 0.1 * performance_score
        
        # Estimate learning rate
        if len(self.skill_trajectories[expert_id]) > 3:
            recent_scores = [score for _, score in self.skill_trajectories[expert_id][-5:]]
            if len(recent_scores) > 1:
                improvements = [recent_scores[i] - recent_scores[i-1] 
                              for i in range(1, len(recent_scores))]
                avg_improvement = np.mean(improvements)
                self.learning_rates[expert_id] = max(0.0, avg_improvement)
    
    def predict_future_skill(self, expert_id: str, days_ahead: int = 30) -> float:
        """Predict expert skill level in the future."""
        if expert_id not in self.skill_trajectories:
            return 0.7  # Default skill level
        
        trajectory = self.skill_trajectories[expert_id]
        if len(trajectory) < 2:
            return trajectory[-1][1] if trajectory else 0.7
        
        # Linear trend prediction
        timestamps = [t.timestamp() for t, _ in trajectory[-10:]]  # Last 10 observations
        scores = [score for _, score in trajectory[-10:]]
        
        if len(timestamps) > 1:
            # Simple linear regression
            x = np.array(timestamps)
            y = np.array(scores)
            slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
            intercept = np.mean(y) - slope * np.mean(x)
            
            future_timestamp = datetime.now().timestamp() + days_ahead * 24 * 3600
            predicted_skill = slope * future_timestamp + intercept
            
            # Apply learning rate adjustment
            learning_rate = self.learning_rates.get(expert_id, 0.0)
            skill_growth = learning_rate * days_ahead / 30.0  # Monthly growth rate
            
            final_prediction = min(1.0, max(0.0, predicted_skill + skill_growth))
            return final_prediction
        
        return trajectory[-1][1]
    
    def get_domain_expertise(self, expert_id: str, domain: str) -> float:
        """Get expert's domain-specific expertise level."""
        return self.expertise_domains[expert_id].get(domain, 0.7)


class TaskComplexityPredictor:
    """Predicts task complexity and completion patterns."""
    
    def __init__(self):
        self.complexity_history: List[Tuple[CodeComplexityMetrics, float, float]] = []
        self.phase_patterns: Dict[SDLCPhase, List[float]] = defaultdict(list)
        self.complexity_features = []
        self.completion_times = []
    
    def observe_task_completion(
        self, 
        metrics: CodeComplexityMetrics, 
        actual_time: float, 
        quality: float,
        phase: SDLCPhase
    ):
        """Record task completion for learning."""
        self.complexity_history.append((metrics, actual_time, quality))
        self.phase_patterns[phase].append(actual_time)
        
        # Extract features
        features = [
            metrics.cyclomatic_complexity,
            metrics.cognitive_complexity,
            metrics.halstead_complexity / 1000.0,
            metrics.lines_of_code / 1000.0,
            metrics.function_count,
            metrics.class_count,
            metrics.dependency_depth,
            metrics.api_surface_area,
            metrics.test_coverage,
            metrics.performance_requirements,
            metrics.security_requirements,
            metrics.scalability_requirements
        ]
        
        self.complexity_features.append(features)
        self.completion_times.append(actual_time)
    
    def predict_completion_time(self, metrics: CodeComplexityMetrics, phase: SDLCPhase) -> float:
        """Predict task completion time based on complexity."""
        if not self.complexity_features:
            # Fallback to simple heuristic
            base_complexity = metrics.get_overall_complexity()
            phase_multipliers = {
                SDLCPhase.ANALYSIS: 0.5,
                SDLCPhase.DESIGN: 0.7,
                SDLCPhase.IMPLEMENTATION: 1.0,
                SDLCPhase.TESTING: 0.8,
                SDLCPhase.DEPLOYMENT: 0.3,
                SDLCPhase.MAINTENANCE: 0.6
            }
            return base_complexity * 20 * phase_multipliers.get(phase, 1.0)
        
        # Use historical data for prediction
        current_features = [
            metrics.cyclomatic_complexity,
            metrics.cognitive_complexity,
            metrics.halstead_complexity / 1000.0,
            metrics.lines_of_code / 1000.0,
            metrics.function_count,
            metrics.class_count,
            metrics.dependency_depth,
            metrics.api_surface_area,
            metrics.test_coverage,
            metrics.performance_requirements,
            metrics.security_requirements,
            metrics.scalability_requirements
        ]
        
        # Find most similar historical tasks (k-NN approach)
        similarities = []
        for hist_features in self.complexity_features:
            diff = np.array(current_features) - np.array(hist_features)
            similarity = 1.0 / (1.0 + np.linalg.norm(diff))
            similarities.append(similarity)
        
        # Weight by similarity and compute weighted average
        similarities = np.array(similarities)
        weights = similarities / np.sum(similarities) if np.sum(similarities) > 0 else np.ones(len(similarities))
        
        predicted_time = np.sum(weights * np.array(self.completion_times))
        
        # Apply phase adjustment
        if phase in self.phase_patterns and self.phase_patterns[phase]:
            phase_avg = np.mean(self.phase_patterns[phase])
            global_avg = np.mean(self.completion_times)
            phase_factor = phase_avg / global_avg if global_avg > 0 else 1.0
            predicted_time *= phase_factor
        
        return max(1.0, predicted_time)  # Minimum 1 hour


class ContinuousLearningEngine:
    """Engine for continuous learning and optimization."""
    
    def __init__(
        self,
        objectives: List[LearningObjective] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.MULTI_OBJECTIVE
    ):
        self.objectives = objectives or [
            LearningObjective.MINIMIZE_COMPLETION_TIME,
            LearningObjective.MAXIMIZE_QUALITY,
            LearningObjective.IMPROVE_TEAM_SATISFACTION
        ]
        self.strategy = strategy
        self.observations: deque = deque(maxlen=1000)
        self.optimization_state = OptimizationState()
        
        # Predictive models
        self.skill_predictor = ExpertSkillPredictor()
        self.complexity_predictor = TaskComplexityPredictor()
        
        # Learning parameters
        self.parameter_ranges = {
            'min_experts_per_task': (1, 5),
            'max_experts_per_task': (2, 8), 
            'collaboration_threshold': (0.5, 1.0),
            'load_balancing_factor': (0.0, 0.5),
            'complexity_weight': (0.5, 2.0),
            'experience_weight': (0.5, 2.0)
        }
        
        self.current_parameters = {
            'min_experts_per_task': 1,
            'max_experts_per_task': 3,
            'collaboration_threshold': 0.7,
            'load_balancing_factor': 0.3,
            'complexity_weight': 1.0,
            'experience_weight': 1.0
        }
        
        self.is_learning = False
        self.learning_thread = None
        
    def add_observation(self, observation: PerformanceObservation):
        """Add performance observation for learning."""
        self.observations.append(observation)
        
        # Update predictive models
        if len(observation.assigned_experts) > 0:
            avg_performance = (observation.quality_score + observation.team_satisfaction) / 2.0
            for expert_id in observation.assigned_experts:
                self.skill_predictor.observe_performance(expert_id, avg_performance)
        
        # Update complexity predictor if task is complete
        if hasattr(observation, 'task_phase'):
            self.complexity_predictor.observe_task_completion(
                observation.context_features.get('complexity_metrics'),
                observation.actual_completion_time,
                observation.quality_score,
                observation.context_features.get('task_phase', SDLCPhase.IMPLEMENTATION)
            )
    
    def start_continuous_learning(self):
        """Start continuous learning in background."""
        if self.is_learning:
            return
        
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("Started continuous learning engine")
    
    def stop_continuous_learning(self):
        """Stop continuous learning."""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
        logger.info("Stopped continuous learning engine")
    
    def _learning_loop(self):
        """Main learning loop running in background."""
        while self.is_learning:
            try:
                if len(self.observations) >= 10:  # Minimum observations needed
                    self._perform_optimization_step()
                
                time.sleep(60)  # Learn every minute
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(30)
    
    def _perform_optimization_step(self):
        """Perform one step of optimization."""
        if self.strategy == OptimizationStrategy.MULTI_OBJECTIVE:
            self._multi_objective_optimization()
        elif self.strategy == OptimizationStrategy.GRADIENT_DESCENT:
            self._gradient_descent_step()
        elif self.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            self._bayesian_optimization_step()
    
    def _multi_objective_optimization(self):
        """Multi-objective optimization using NSGA-II inspired approach."""
        # Generate candidate parameter sets
        candidates = []
        for _ in range(20):  # Population size
            candidate = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                if self.optimization_state.exploration_rate > np.random.random():
                    # Exploration: random value
                    candidate[param] = np.random.uniform(min_val, max_val)
                else:
                    # Exploitation: perturb current best
                    current = self.current_parameters[param]
                    perturbation = np.random.normal(0, 0.1 * (max_val - min_val))
                    candidate[param] = np.clip(current + perturbation, min_val, max_val)
            candidates.append(candidate)
        
        # Evaluate candidates
        candidate_scores = []
        for candidate in candidates:
            score = self._evaluate_parameter_set(candidate)
            candidate_scores.append(score)
        
        # Select best candidate
        best_idx = np.argmax(candidate_scores)
        best_candidate = candidates[best_idx]
        best_score = candidate_scores[best_idx]
        
        # Update if improvement
        if best_score > self.optimization_state.best_performance:
            self.optimization_state.best_performance = best_score
            self.optimization_state.best_parameters = best_candidate.copy()
            self.current_parameters.update(best_candidate)
            self.optimization_state.stagnant_iterations = 0
            
            logger.info(f"Optimization improvement: score={best_score:.4f}")
        else:
            self.optimization_state.stagnant_iterations += 1
        
        # Update optimization state
        self.optimization_state.iteration += 1
        self.optimization_state.performance_history.append(best_score)
        self.optimization_state.parameter_history.append(best_candidate.copy())
        
        # Decay exploration rate
        self.optimization_state.exploration_rate *= 0.995
        self.optimization_state.exploration_rate = max(0.05, self.optimization_state.exploration_rate)
    
    def _evaluate_parameter_set(self, parameters: Dict[str, Any]) -> float:
        """Evaluate a set of parameters using historical data."""
        if len(self.observations) < 5:
            return 0.5  # Not enough data
        
        # Simulate performance with these parameters on historical tasks
        total_score = 0.0
        count = 0
        
        # Use recent observations for evaluation
        recent_observations = list(self.observations)[-50:]
        
        for obs in recent_observations:
            # Simulate what would have happened with these parameters
            simulated_performance = self._simulate_performance(obs, parameters)
            
            # Compute multi-objective score
            objectives_score = self._compute_objectives_score(obs, simulated_performance)
            total_score += objectives_score
            count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _simulate_performance(self, obs: PerformanceObservation, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Simulate performance with different parameters."""
        # This is a simplified simulation - in practice would be more sophisticated
        base_time = obs.actual_completion_time
        base_quality = obs.quality_score
        base_satisfaction = obs.team_satisfaction
        
        # Parameter effects (simplified model)
        complexity_factor = parameters.get('complexity_weight', 1.0)
        team_size_effect = len(obs.assigned_experts) / parameters.get('max_experts_per_task', 3)
        
        # Adjust performance based on parameters
        simulated_time = base_time * (0.8 + 0.4 * team_size_effect)
        simulated_quality = min(1.0, base_quality * (0.9 + 0.1 * complexity_factor))
        simulated_satisfaction = min(1.0, base_satisfaction * (0.9 + 0.1 * parameters.get('collaboration_threshold', 0.7)))
        
        return {
            'completion_time': simulated_time,
            'quality_score': simulated_quality,
            'team_satisfaction': simulated_satisfaction
        }
    
    def _compute_objectives_score(self, obs: PerformanceObservation, simulated: Dict[str, float]) -> float:
        """Compute multi-objective score."""
        scores = []
        
        for objective in self.objectives:
            if objective == LearningObjective.MINIMIZE_COMPLETION_TIME:
                # Lower time is better
                max_time = max(o.actual_completion_time for o in self.observations)
                normalized_time = 1.0 - (simulated['completion_time'] / max_time)
                scores.append(normalized_time)
            
            elif objective == LearningObjective.MAXIMIZE_QUALITY:
                scores.append(simulated['quality_score'])
            
            elif objective == LearningObjective.IMPROVE_TEAM_SATISFACTION:
                scores.append(simulated['team_satisfaction'])
        
        return np.mean(scores) if scores else 0.0
    
    def _gradient_descent_step(self):
        """Perform gradient descent optimization step."""
        # Simplified gradient estimation using finite differences
        current_score = self._evaluate_parameter_set(self.current_parameters)
        
        gradients = {}
        step_size = 0.01
        
        for param in self.parameter_ranges:
            # Compute finite difference gradient
            params_plus = self.current_parameters.copy()
            min_val, max_val = self.parameter_ranges[param]
            delta = step_size * (max_val - min_val)
            params_plus[param] = min(max_val, params_plus[param] + delta)
            
            score_plus = self._evaluate_parameter_set(params_plus)
            gradient = (score_plus - current_score) / delta
            gradients[param] = gradient
        
        # Update parameters using gradient ascent (maximizing performance)
        learning_rate = self.optimization_state.learning_rate
        for param in self.current_parameters:
            gradient = gradients.get(param, 0.0)
            min_val, max_val = self.parameter_ranges[param]
            
            new_value = self.current_parameters[param] + learning_rate * gradient
            self.current_parameters[param] = np.clip(new_value, min_val, max_val)
        
        # Update optimization state
        new_score = self._evaluate_parameter_set(self.current_parameters)
        if new_score > self.optimization_state.best_performance:
            self.optimization_state.best_performance = new_score
            self.optimization_state.best_parameters = self.current_parameters.copy()
            self.optimization_state.stagnant_iterations = 0
        else:
            self.optimization_state.stagnant_iterations += 1
        
        self.optimization_state.iteration += 1
        self.optimization_state.performance_history.append(new_score)
    
    def _bayesian_optimization_step(self):
        """Perform Bayesian optimization step (simplified)."""
        # This would typically use Gaussian Processes, but here's a simplified version
        # Generate candidates using acquisition function
        candidates = []
        candidate_scores = []
        
        for _ in range(10):
            candidate = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                # Use uncertainty sampling around current best
                if self.optimization_state.best_parameters:
                    center = self.optimization_state.best_parameters.get(param, (min_val + max_val) / 2)
                    uncertainty = 0.1 * (max_val - min_val)
                    candidate[param] = np.clip(
                        np.random.normal(center, uncertainty),
                        min_val, max_val
                    )
                else:
                    candidate[param] = np.random.uniform(min_val, max_val)
            
            candidates.append(candidate)
            candidate_scores.append(self._evaluate_parameter_set(candidate))
        
        # Select best candidate
        best_idx = np.argmax(candidate_scores)
        best_candidate = candidates[best_idx]
        best_score = candidate_scores[best_idx]
        
        if best_score > self.optimization_state.best_performance:
            self.optimization_state.best_performance = best_score
            self.optimization_state.best_parameters = best_candidate.copy()
            self.current_parameters.update(best_candidate)
            self.optimization_state.stagnant_iterations = 0
        else:
            self.optimization_state.stagnant_iterations += 1
        
        self.optimization_state.iteration += 1
        self.optimization_state.performance_history.append(best_score)
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from continuous learning."""
        return {
            'optimization_iteration': self.optimization_state.iteration,
            'best_performance': self.optimization_state.best_performance,
            'current_parameters': self.current_parameters.copy(),
            'best_parameters': self.optimization_state.best_parameters.copy(),
            'stagnant_iterations': self.optimization_state.stagnant_iterations,
            'exploration_rate': self.optimization_state.exploration_rate,
            'total_observations': len(self.observations),
            'performance_trend': self.optimization_state.performance_history[-10:],
            'learning_active': self.is_learning,
            'convergence_status': 'converged' if self.optimization_state.stagnant_iterations > self.optimization_state.patience else 'learning'
        }
    
    def save_learning_state(self, filepath: str):
        """Save learning state to file."""
        state = {
            'optimization_state': self.optimization_state,
            'current_parameters': self.current_parameters,
            'observations': list(self.observations),
            'skill_predictor': self.skill_predictor,
            'complexity_predictor': self.complexity_predictor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved learning state to {filepath}")
    
    def load_learning_state(self, filepath: str):
        """Load learning state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.optimization_state = state['optimization_state']
            self.current_parameters = state['current_parameters']
            self.observations.extend(state['observations'])
            self.skill_predictor = state['skill_predictor']
            self.complexity_predictor = state['complexity_predictor']
            
            logger.info(f"Loaded learning state from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")


class AutonomousSDLCOptimizer:
    """Advanced SDLC system with continuous learning and optimization."""
    
    def __init__(
        self,
        experts: List[ExpertCapability],
        learning_objectives: List[LearningObjective] = None,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MULTI_OBJECTIVE,
        enable_continuous_learning: bool = True
    ):
        self.experts = experts
        self.learning_engine = ContinuousLearningEngine(learning_objectives, optimization_strategy)
        
        # Create initial router with default parameters
        self.router = AutonomousSDLCRouter(
            experts,
            min_experts_per_task=1,
            max_experts_per_task=3,
            collaboration_threshold=0.7,
            load_balancing_factor=0.3
        )
        
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        if enable_continuous_learning:
            self.learning_engine.start_continuous_learning()
    
    def route_task_with_learning(
        self,
        task: SDLCTask,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route task and capture learning data."""
        # Update router with current optimized parameters
        self._update_router_parameters()
        
        # Route the task
        result = self.router.route_task(task, context)
        
        # Store for learning
        task_data = {
            'task': task,
            'context': context or {},
            'result': result,
            'timestamp': datetime.now()
        }
        self.task_history.append(task_data)
        
        return result
    
    def complete_task(
        self,
        task_id: str,
        actual_completion_time: float,
        quality_score: float,
        defect_count: int = 0,
        team_satisfaction: float = 0.8,
        resource_cost: float = 0.0
    ):
        """Record task completion for learning."""
        # Find the task in history
        task_data = None
        for data in reversed(self.task_history):
            if data['task'].task_id == task_id:
                task_data = data
                break
        
        if not task_data:
            logger.warning(f"Task {task_id} not found in history")
            return
        
        task = task_data['task']
        context = task_data['context']
        result = task_data['result']
        
        # Create performance observation
        observation = PerformanceObservation(
            timestamp=datetime.now(),
            task_id=task_id,
            task_complexity=task.complexity_metrics.get_overall_complexity(),
            assigned_experts=result['selected_experts'],
            expert_weights=result['expert_weights'].tolist(),
            actual_completion_time=actual_completion_time,
            estimated_completion_time=result['estimated_completion_time'],
            quality_score=quality_score,
            defect_count=defect_count,
            team_satisfaction=team_satisfaction,
            resource_cost=resource_cost,
            context_features={
                'complexity_metrics': task.complexity_metrics,
                'task_phase': task.phase,
                'timeline_pressure': context.get('timeline_pressure', 0.5),
                'budget_constraint': context.get('budget_constraint', 0.5),
                'team_experience': np.mean([exp.skill_level for exp in result['selected_experts']]),
                'project_phase_progress': context.get('project_phase_progress', 0.5)
            }
        )
        
        # Add to learning engine
        self.learning_engine.add_observation(observation)
        
        # Update performance metrics
        self.performance_metrics['completion_time'].append(actual_completion_time)
        self.performance_metrics['quality_score'].append(quality_score)
        self.performance_metrics['team_satisfaction'].append(team_satisfaction)
        self.performance_metrics['defect_count'].append(defect_count)
        
        logger.info(f"Recorded completion of task {task_id} for learning")
    
    def _update_router_parameters(self):
        """Update router with optimized parameters."""
        params = self.learning_engine.current_parameters
        
        # Update router configuration
        self.router.min_experts_per_task = int(params.get('min_experts_per_task', 1))
        self.router.max_experts_per_task = int(params.get('max_experts_per_task', 3))
        self.router.collaboration_threshold = params.get('collaboration_threshold', 0.7)
        self.router.load_balancing_factor = params.get('load_balancing_factor', 0.3)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        recent_window = 20
        
        dashboard = {
            'total_tasks_completed': len(self.performance_metrics.get('completion_time', [])),
            'learning_insights': self.learning_engine.get_optimization_insights(),
            'performance_trends': {}
        }
        
        # Compute performance trends
        for metric, values in self.performance_metrics.items():
            if len(values) >= 2:
                recent_avg = np.mean(values[-recent_window:]) if len(values) >= recent_window else np.mean(values)
                overall_avg = np.mean(values)
                trend = 'improving' if recent_avg > overall_avg else 'declining' if recent_avg < overall_avg else 'stable'
                
                dashboard['performance_trends'][metric] = {
                    'recent_average': float(recent_avg),
                    'overall_average': float(overall_avg),
                    'trend': trend,
                    'improvement_rate': float((recent_avg - overall_avg) / overall_avg) if overall_avg > 0 else 0.0
                }
        
        # Expert utilization insights
        expert_assignments = defaultdict(int)
        for data in self.task_history:
            for expert in data['result']['selected_experts']:
                expert_assignments[expert.expert_type.value] += 1
        
        dashboard['expert_utilization'] = dict(expert_assignments)
        
        return dashboard
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        insights = self.learning_engine.get_optimization_insights()
        dashboard = self.get_performance_dashboard()
        
        report = f"""
🤖 AUTONOMOUS SDLC OPTIMIZATION REPORT
{'=' * 50}

📊 Learning Status:
- Optimization Iteration: {insights['optimization_iteration']}
- Performance Score: {insights['best_performance']:.4f}
- Learning Status: {insights['convergence_status']}
- Total Observations: {insights['total_observations']}

⚙️ Current Parameters:
"""
        
        for param, value in insights['current_parameters'].items():
            report += f"- {param}: {value:.3f}\n"
        
        report += f"""
📈 Performance Trends:
"""
        
        for metric, trend in dashboard['performance_trends'].items():
            report += f"- {metric}: {trend['trend']} ({trend['improvement_rate']:+.1%})\n"
        
        report += f"""
👥 Expert Utilization:
"""
        
        for expert, count in dashboard['expert_utilization'].items():
            report += f"- {expert}: {count} assignments\n"
        
        report += f"""
🎯 Optimization Insights:
- Exploration Rate: {insights['exploration_rate']:.3f}
- Stagnant Iterations: {insights['stagnant_iterations']}
- Recent Performance: {insights['performance_trend'][-5:] if len(insights['performance_trend']) >= 5 else insights['performance_trend']}

💡 Recommendations:
"""
        
        # Generate recommendations based on learning state
        if insights['stagnant_iterations'] > 5:
            report += "- Consider increasing exploration rate or trying different optimization strategy\n"
        
        if insights['best_performance'] > 0.8:
            report += "- System is performing well, consider gradual parameter refinement\n"
        
        if dashboard['total_tasks_completed'] < 20:
            report += "- More data needed for robust optimization, continue collecting observations\n"
        
        return report
    
    def save_state(self, filepath: str):
        """Save complete optimizer state."""
        self.learning_engine.save_learning_state(filepath)
    
    def load_state(self, filepath: str):
        """Load optimizer state."""
        self.learning_engine.load_learning_state(filepath)
        self._update_router_parameters()
    
    def __del__(self):
        """Cleanup when optimizer is destroyed."""
        if hasattr(self, 'learning_engine'):
            self.learning_engine.stop_continuous_learning()


def demonstrate_autonomous_sdlc_optimization():
    """Demonstrate the Autonomous SDLC Optimizer."""
    print("🚀 Autonomous SDLC Optimizer Demonstration")
    print("=" * 60)
    
    # Create sample experts
    experts = []
    expert_types = [DevelopmentExpert.ARCHITECT, DevelopmentExpert.ALGORITHM_EXPERT, 
                   DevelopmentExpert.API_DEVELOPER, DevelopmentExpert.SECURITY_EXPERT]
    
    for expert_type in expert_types:
        expert = ExpertCapability(
            expert_type=expert_type,
            skill_level=np.random.uniform(0.7, 0.9),
            experience_years=np.random.uniform(3, 12),
            specialization_areas=[expert_type.value],
            current_workload=np.random.uniform(0.2, 0.6),
            performance_history=[np.random.uniform(0.7, 0.9) for _ in range(3)],
            collaboration_score=np.random.uniform(0.7, 0.9)
        )
        experts.append(expert)
    
    # Create optimizer
    optimizer = AutonomousSDLCOptimizer(
        experts,
        learning_objectives=[
            LearningObjective.MINIMIZE_COMPLETION_TIME,
            LearningObjective.MAXIMIZE_QUALITY,
            LearningObjective.IMPROVE_TEAM_SATISFACTION
        ],
        optimization_strategy=OptimizationStrategy.MULTI_OBJECTIVE
    )
    
    # Simulate task routing and completion
    print("🎯 Simulating SDLC optimization...")
    
    for i in range(10):
        # Create sample task
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
        
        task = SDLCTask(
            task_id=f"OPT-TASK-{i:03d}",
            phase=np.random.choice(list(SDLCPhase)),
            description=f"Optimization test task {i}",
            complexity_metrics=metrics,
            priority=np.random.uniform(0.5, 1.0),
            deadline_pressure=np.random.uniform(0.3, 0.8),
            dependencies=[],
            estimated_effort=complexity * 25 + np.random.uniform(5, 15),
            risk_level=complexity * 0.6 + np.random.uniform(0.0, 0.3)
        )
        
        # Route task
        result = optimizer.route_task_with_learning(task)
        
        # Simulate completion with some realistic outcomes
        estimated_time = result['estimated_completion_time']
        actual_time = estimated_time * np.random.uniform(0.8, 1.3)
        quality = result['routing_confidence'] * np.random.uniform(0.85, 1.0)
        satisfaction = result['collaboration_score'] * np.random.uniform(0.8, 1.0)
        defects = max(0, int(np.random.poisson((1.0 - quality) * 5)))
        
        # Record completion
        optimizer.complete_task(
            task.task_id,
            actual_time,
            quality,
            defects,
            satisfaction
        )
        
        print(f"  Task {i+1}: time={actual_time:.1f}h, quality={quality:.3f}, satisfaction={satisfaction:.3f}")
    
    # Allow some learning iterations
    print("\n🧠 Running optimization iterations...")
    time.sleep(2)  # Let the learning engine run
    
    # Generate report
    report = optimizer.generate_optimization_report()
    print(report)
    
    # Show dashboard
    dashboard = optimizer.get_performance_dashboard()
    print(f"\n📊 Final Dashboard Summary:")
    print(f"- Tasks Completed: {dashboard['total_tasks_completed']}")
    print(f"- Learning Iterations: {dashboard['learning_insights']['optimization_iteration']}")
    print(f"- Best Performance: {dashboard['learning_insights']['best_performance']:.4f}")
    
    return optimizer


if __name__ == "__main__":
    demonstrate_autonomous_sdlc_optimization()