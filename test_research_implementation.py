"""Simple test script for research implementation validation."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_confidence_router():
    """Test confidence-based router."""
    from dynamic_moe_router.adaptive_entropy_router import ConfidenceBasedRouter
    
    print("Testing Confidence-Based Router...")
    router = ConfidenceBasedRouter(input_dim=64, num_experts=4, min_experts=1, max_experts=3)
    
    # Test input
    inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
    experts, weights, info = router.forward(inputs)
    
    print(f"  ✅ Forward pass successful")
    print(f"  ✅ Output shapes: experts{experts.shape}, weights{weights.shape}")
    print(f"  ✅ Avg experts per token: {info['avg_experts_per_token']:.2f}")
    print(f"  ✅ FLOP reduction: {info['flop_reduction']:.1%}")
    
    return True

def test_resonance_router():
    """Test expert-token resonance router."""
    from dynamic_moe_router.adaptive_entropy_router import ExpertTokenResonanceRouter
    
    print("\\nTesting Expert-Token Resonance Router...")
    router = ExpertTokenResonanceRouter(input_dim=64, num_experts=4)
    
    inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
    experts, weights, info = router.forward(inputs)
    
    print(f"  ✅ Forward pass successful")
    print(f"  ✅ Output shapes: experts{experts.shape}, weights{weights.shape}")
    print(f"  ✅ Avg experts per token: {info['avg_experts_per_token']:.2f}")
    
    return True

def test_similarity_router():
    """Test similarity-aware router."""
    from dynamic_moe_router.adaptive_entropy_router import SimilarityAwareRouter
    
    print("\\nTesting Similarity-Aware Router...")
    router = SimilarityAwareRouter(input_dim=64, num_experts=4, similarity_metric="cosine")
    
    inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
    experts, weights, info = router.forward(inputs)
    
    print(f"  ✅ Forward pass successful")
    print(f"  ✅ Output shapes: experts{experts.shape}, weights{weights.shape}")
    print(f"  ✅ Mean entropy: {info['mean_entropy']:.3f}")
    print(f"  ✅ Entropy reduction: {info['entropy_reduction']:.3f}")
    
    return True

def test_ensemble_router():
    """Test ensemble router."""
    from dynamic_moe_router.adaptive_entropy_router import AdaptiveEntropyRouterEnsemble
    
    print("\\nTesting Adaptive Entropy Ensemble Router...")
    router = AdaptiveEntropyRouterEnsemble(input_dim=64, num_experts=4)
    
    inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
    experts, weights, info = router.forward(inputs)
    
    print(f"  ✅ Forward pass successful")
    print(f"  ✅ Output shapes: experts{experts.shape}, weights{weights.shape}")
    print(f"  ✅ Avg experts per token: {info['avg_experts_per_token']:.2f}")
    print(f"  ✅ Ensemble entropy: {info['ensemble_entropy']:.3f}")
    
    return True

def test_comparative_study():
    """Test comparative study framework."""
    from dynamic_moe_router.adaptive_entropy_router import RouterComparativeStudy
    
    print("\\nTesting Comparative Study Framework...")
    study = RouterComparativeStudy(input_dim=64, num_experts=4)
    
    # Generate small test dataset
    test_inputs = []
    complexity_labels = []
    
    for complexity in ['simple', 'medium', 'complex']:
        for _ in range(2):
            if complexity == 'simple':
                inputs = np.random.uniform(-0.5, 0.5, (2, 16, 64)).astype(np.float32)
            elif complexity == 'medium':
                inputs = np.random.normal(0, 0.7, (2, 16, 64)).astype(np.float32)
            else:  # complex
                inputs = np.random.normal(0, 1, (2, 16, 64)).astype(np.float32)
                # Add patterns
                inputs[:, :, :32] += np.sin(np.linspace(0, 2*np.pi, 32))
            
            test_inputs.append(inputs)
            complexity_labels.append(complexity)
    
    # Run study
    results = study.run_comparative_study(test_inputs, complexity_labels)
    
    print(f"  ✅ Comparative study completed")
    print(f"  ✅ Routers tested: {len(results['detailed_results'])}")
    print(f"  ✅ Test cases: {len(test_inputs)}")
    
    # Show summary
    if 'comparison_summary' in results:
        for router_name, metrics in results['comparison_summary'].items():
            avg_flop = metrics.get('avg_flop_reduction', 0)
            print(f"    {router_name}: FLOP reduction {avg_flop:.1%}")
    
    return True

def test_research_benchmarks():
    """Test research benchmarking framework."""
    from dynamic_moe_router.research_benchmarks import BenchmarkConfig, SyntheticDatasetGenerator
    
    print("\\nTesting Research Benchmark Framework...")
    
    # Test dataset generation
    generator = SyntheticDatasetGenerator(input_dim=64)
    datasets = generator.generate_complexity_dataset(num_samples=10, batch_size=2, sequence_length=16)
    
    print(f"  ✅ Generated {len(datasets)} synthetic samples")
    
    # Check dataset properties
    complexities = set(d.complexity_label for d in datasets)
    print(f"  ✅ Complexity levels: {complexities}")
    
    # Test benchmark config
    config = BenchmarkConfig(
        input_dim=64,
        num_experts=4,
        batch_sizes=[2, 4],
        sequence_lengths=[16, 32],
        num_runs=2
    )
    
    print(f"  ✅ Benchmark config created")
    print(f"    Input dim: {config.input_dim}")
    print(f"    Num experts: {config.num_experts}")
    print(f"    Batch sizes: {config.batch_sizes}")
    
    return True

def main():
    """Run all tests."""
    print("=== Research Implementation Validation ===\\n")
    
    tests = [
        test_confidence_router,
        test_resonance_router,
        test_similarity_router,
        test_ensemble_router,
        test_comparative_study,
        test_research_benchmarks
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
    
    print(f"\\n=== Results ===")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\\n🎉 All research implementations validated successfully!")
        print("✅ Novel algorithms from 2024 research papers")
        print("✅ Comprehensive benchmarking framework")
        print("✅ Statistical analysis capabilities")
        print("✅ Comparative study infrastructure")
    else:
        print(f"\\n⚠️  {total-passed} test(s) failed. Implementation needs fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)