# -*- coding: utf-8 -*-
"""
Pre-flight

Tests:
1. API connection works
2. SameJudge returns expected results on known pairs
3. Rate limiting doesn't kill us
4. Merge logic picks correct canonical

"""
"""
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test pairs - DIFFERENT from few-shot examples to test generalization
# Few-shot now includes: GDPR, USA, AI, EU AI Act, EU, ML
TEST_PAIRS = [
    # YES - same entity (NOT in few-shot examples)
    {
        'entity1': {'name': 'NLP', 'type': 'Technology', 'description': 'natural language processing field'},
        'entity2': {'name': 'natural language processing', 'type': 'Technology', 'description': 'AI subfield for text'},
        'expected': True,
        'reason': 'same concept, abbreviation'
    },
    {
        'entity1': {'name': 'Germany', 'type': 'Location', 'description': 'European country'},
        'entity2': {'name': 'Federal Republic of Germany', 'type': 'Location', 'description': 'country in central Europe'},
        'expected': True,
        'reason': 'same country'
    },
    {
        'entity1': {'name': 'HIPAA', 'type': 'Regulation', 'description': 'US healthcare privacy law'},
        'entity2': {'name': 'Health Insurance Portability and Accountability Act', 'type': 'Regulation', 'description': 'American health data law'},
        'expected': True,
        'reason': 'same law, abbreviation'
    },
    # NO - antonyms/opposing
    {
        'entity1': {'name': 'security', 'type': 'Risk', 'description': 'protection from threats'},
        'entity2': {'name': 'threat', 'type': 'Risk', 'description': 'potential danger'},
        'expected': False,
        'reason': 'antonyms - security vs threat'
    },
    {
        'entity1': {'name': 'opportunities', 'type': 'EconomicConcept', 'description': 'favorable circumstances'},
        'entity2': {'name': 'challenges', 'type': 'EconomicConcept', 'description': 'difficulties to overcome'},
        'expected': False,
        'reason': 'antonyms - opportunities vs challenges'
    },
    # NO - X vs X-issues pattern
    {
        'entity1': {'name': 'transparency', 'type': 'RegulatoryConcept', 'description': 'openness and clarity'},
        'entity2': {'name': 'transparency concerns', 'type': 'RegulatoryConcept', 'description': 'worries about lack of transparency'},
        'expected': False,
        'reason': 'concept vs concerns about concept'
    },
    {
        'entity1': {'name': 'ethics', 'type': 'PoliticalConcept', 'description': 'moral principles'},
        'entity2': {'name': 'ethical issues', 'type': 'PoliticalConcept', 'description': 'moral problems'},
        'expected': False,
        'reason': 'concept vs issues'
    },
    # NO - specific vs generic
    {
        'entity1': {'name': 'HIPAA', 'type': 'Regulation', 'description': 'US healthcare privacy law'},
        'entity2': {'name': 'health regulations', 'type': 'Regulation', 'description': 'laws about healthcare'},
        'expected': False,
        'reason': 'specific law vs generic category'
    },
    # NO - different identifiers
    {
        'entity1': {'name': 'Section 3', 'type': 'DocumentSection', 'description': 'third section'},
        'entity2': {'name': 'Section 4', 'type': 'DocumentSection', 'description': 'fourth section'},
        'expected': False,
        'reason': 'different sections'
    },
    # NO - related technologies
    {
        'entity1': {'name': 'neural networks', 'type': 'Technology', 'description': 'brain-inspired computing'},
        'entity2': {'name': 'deep learning', 'type': 'Technology', 'description': 'multi-layer neural networks'},
        'expected': False,
        'reason': 'related but different technologies'
    },
]


def test_samejudge():
    """Test SameJudge on known pairs."""
    from src.processing.entities.semantic_disambiguator import SameJudge
    
    print("\n" + "="*60)
    print("SAMEJUDGE PRE-FLIGHT TEST")
    print("="*60)
    
    # Initialize
    print("\n[1] Initializing SameJudge (Mistral-7B + RateLimiter)...")
    try:
        judge = SameJudge()
        print("    ✓ API connection OK")
        print("    ✓ RateLimiter initialized (2900 RPM)")
    except Exception as e:
        print(f"    ✗ Initialization FAILED: {e}")
        return False
    
    # Test pairs
    print(f"\n[2] Testing {len(TEST_PAIRS)} pairs...")
    correct = 0
    times = []
    
    for i, test in enumerate(TEST_PAIRS, 1):
        start = time.time()
        is_same, reasoning = judge.judge_pair(test['entity1'], test['entity2'])
        elapsed = time.time() - start
        times.append(elapsed)
        
        expected = test['expected']
        match = is_same == expected
        
        status = "✓" if match else "✗"
        expected_str = "SAME" if expected else "DIFF"
        actual_str = "SAME" if is_same else "DIFF"
        
        print(f"    [{i}] {status} {test['entity1']['name'][:25]:<25} vs {test['entity2']['name'][:25]:<25}")
        print(f"        Expected: {expected_str}, Got: {actual_str} ({elapsed:.2f}s) - {test['reason']}")
        
        if match:
            correct += 1
    
    # Summary
    accuracy = correct / len(TEST_PAIRS) * 100
    avg_time = sum(times) / len(times)
    
    print(f"\n[3] Results:")
    print(f"    Accuracy:  {correct}/{len(TEST_PAIRS)} ({accuracy:.0f}%)")
    print(f"    Avg time:  {avg_time:.2f}s per pair")
    print(f"    Est. time: {avg_time * 21000 / 3600:.1f}h for 21K pairs")
    print(f"    Est. cost: ${21000 * 0.0001:.2f}")
    
    # Rate limit test
    print(f"\n[4] Rate limit test (10 rapid calls)...")
    start = time.time()
    for _ in range(10):
        judge.judge_pair(TEST_PAIRS[0]['entity1'], TEST_PAIRS[0]['entity2'])
    burst_time = time.time() - start
    print(f"    10 calls in {burst_time:.2f}s ({10/burst_time:.1f} calls/sec)")
    
    # Rate limiter stats
    rl_stats = judge.rate_limiter.get_stats()
    print(f"\n[5] Rate Limiter Stats:")
    print(f"    Total calls: {rl_stats['total_calls']}")
    print(f"    Wait time:   {rl_stats['total_wait_time_sec']}s")
    print(f"    Utilization: {rl_stats['utilization_pct']}%")
    
    if accuracy >= 75:
        print(f"\n{'='*60}")
        print("PRE-FLIGHT PASSED - Safe to run full phase")
        print(f"{'='*60}")
        return True
    else:
        print(f"\n{'='*60}")
        print("PRE-FLIGHT FAILED - Review SameJudge prompt")
        print(f"{'='*60}")
        return False


def test_merge_logic():
    """Test that merge logic picks correct canonical."""
    from src.processing.entities.semantic_disambiguator import apply_merges
    
    print("\n" + "="*60)
    print("MERGE LOGIC TEST")
    print("="*60)
    
    # Create test entities - canonical selection is based on occurrence count in group
    # "EU AI Act" appears 5 times, "European AI Act" 2 times, "AI Act (EU)" 1 time
    entities = []
    for _ in range(5):
        entities.append({'name': 'EU AI Act', 'type': 'Regulation', 'description': 'European Union AI Act'})
    for _ in range(2):
        entities.append({'name': 'European AI Act', 'type': 'Regulation', 'description': 'The European AI Act'})
    entities.append({'name': 'AI Act (EU)', 'type': 'Regulation', 'description': 'EU AI legislation'})
    
    # Merge pairs
    pairs = [
        {'entity1_key': ('EU AI Act', 'Regulation'), 'entity2_key': ('European AI Act', 'Regulation'), 'similarity': 0.99},
        {'entity1_key': ('EU AI Act', 'Regulation'), 'entity2_key': ('AI Act (EU)', 'Regulation'), 'similarity': 0.98},
    ]
    
    result, aliases = apply_merges(entities, pairs)
    
    print(f"\n  Input: {len(entities)} entities (5x 'EU AI Act', 2x 'European AI Act', 1x 'AI Act (EU)')")
    print(f"  Output: {len(result)} entities")
    print(f"  Aliases: {len(aliases)}")
    
    # Check canonical is most frequent
    canonical = result[0]['name'] if result else None
    print(f"\n  Canonical: '{canonical}'")
    print(f"  Expected:  'EU AI Act' (most frequent)")
    
    if canonical == 'EU AI Act':
        print("  ✓ Merge logic correct")
        return True
    else:
        print("  ✗ Merge logic incorrect")
        return False


if __name__ == '__main__':
    import sys
    
    # Run tests
    judge_ok = test_samejudge()
    merge_ok = test_merge_logic()
    
    if judge_ok and merge_ok:
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        print("\nNext step:")
        print("  python -m src.processing.entities.disambiguation_processor --phase full")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("TESTS FAILED - Fix issues before full run")
        print("="*60)
        sys.exit(1)