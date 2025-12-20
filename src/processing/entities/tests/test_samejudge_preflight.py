# -*- coding: utf-8 -*-
"""
Pre-flight test for SameJudge before committing to full 21K pair run.

Tests:
1. API connection works
2. SameJudge returns expected results on known pairs
3. Rate limiting doesn't kill us
4. Merge logic picks correct canonical

Usage:
    python -m src.processing.entities.tests.test_samejudge_preflight
"""

import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test pairs - known SAME and DIFF from threshold analysis
TEST_PAIRS = [
    # SAME pairs (should return YES)
    {
        'entity1': {'name': 'privacy and security', 'type': 'Risk', 'description': 'Data protection concerns'},
        'entity2': {'name': 'security and privacy', 'type': 'Risk', 'description': 'Security and data protection'},
        'expected': True,
        'reason': 'word order swap'
    },
    {
        'entity1': {'name': 'facial recognition technology', 'type': 'Technology', 'description': 'Biometric identification'},
        'entity2': {'name': 'facial recognition technologies', 'type': 'Technology', 'description': 'Face recognition systems'},
        'expected': True,
        'reason': 'singular/plural'
    },
    {
        'entity1': {'name': 'United States', 'type': 'Location', 'description': 'North American country'},
        'entity2': {'name': 'United States of America', 'type': 'Location', 'description': 'USA'},
        'expected': True,
        'reason': 'abbreviation'
    },
    {
        'entity1': {'name': 'General Data Protection Regulation (GDPR)', 'type': 'Regulation', 'description': 'EU data protection law'},
        'entity2': {'name': 'European General Data Protection Regulation (GDPR)', 'type': 'Regulation', 'description': 'EU privacy regulation'},
        'expected': True,
        'reason': 'same regulation, different prefix'
    },
    # DIFF pairs (should return NO)
    {
        'entity1': {'name': 'Turuta O. V.', 'type': 'Organization', 'description': 'Author'},
        'entity2': {'name': 'Turuta O. P.', 'type': 'Organization', 'description': 'Author'},
        'expected': False,
        'reason': 'different people (V vs P)'
    },
    {
        'entity1': {'name': 'April 2025', 'type': 'Location', 'description': 'Date reference'},
        'entity2': {'name': 'March 2025', 'type': 'Location', 'description': 'Date reference'},
        'expected': False,
        'reason': 'different months'
    },
    {
        'entity1': {'name': 'Level 1', 'type': 'RegulatoryConcept', 'description': 'Risk classification tier'},
        'entity2': {'name': 'Level 2', 'type': 'RegulatoryConcept', 'description': 'Risk classification tier'},
        'expected': False,
        'reason': 'different levels'
    },
    {
        'entity1': {'name': 'Data Protection Act', 'type': 'Regulation', 'description': 'UK data law'},
        'entity2': {'name': 'Privacy Act', 'type': 'Regulation', 'description': 'US privacy law'},
        'expected': False,
        'reason': 'different regulations'
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