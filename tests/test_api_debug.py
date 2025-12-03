"""
Minimal API Test Script - Debug Together.ai API Issues

Tests three scenarios:
1. Basic connectivity (hello world)
2. Simple JSON extraction
3. Actual relation extraction (minimal)

Usage:
    python test_api_debug.py
"""

import os
import json
from together import Together
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_basic_connectivity():
    """Test 1: Basic API connectivity"""
    print("\n" + "="*80)
    print("TEST 1: BASIC CONNECTIVITY")
    print("="*80)
    
    try:
        client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello World' and nothing else."}
            ],
            max_tokens=50,
            temperature=0.0
        )
        
        print(f"✓ API call succeeded")
        print(f"  Response type: {type(response)}")
        print(f"  Has choices: {hasattr(response, 'choices')}")
        print(f"  Number of choices: {len(response.choices)}")
        print(f"  Finish reason: {response.choices[0].finish_reason}")
        
        content = response.choices[0].message.content
        print(f"  Content type: {type(content)}")
        print(f"  Content is None: {content is None}")
        print(f"  Content: '{content}'")
        
        if hasattr(response, 'usage'):
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_json():
    """Test 2: Simple JSON extraction"""
    print("\n" + "="*80)
    print("TEST 2: SIMPLE JSON EXTRACTION")
    print("="*80)
    
    prompt = """Extract the name and age from this text:
"John is 25 years old."

OUTPUT FORMAT (JSON only):
{
  "name": "string",
  "age": number
}"""
    
    try:
        client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0
        )
        
        print(f"✓ API call succeeded")
        print(f"  Finish reason: {response.choices[0].finish_reason}")
        
        content = response.choices[0].message.content
        print(f"  Content length: {len(content) if content else 0} chars")
        print(f"  Content: {content}")
        
        if content:
            # Try to parse JSON
            try:
                parsed = json.loads(content.strip().replace("```json", "").replace("```", ""))
                print(f"  ✓ Parsed JSON: {parsed}")
            except json.JSONDecodeError as e:
                print(f"  ✗ JSON parse failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_relation_extraction():
    """Test 3: Minimal relation extraction (2 entities, 1 chunk)"""
    print("\n" + "="*80)
    print("TEST 3: MINIMAL RELATION EXTRACTION")
    print("="*80)
    
    prompt = """You are a knowledge graph construction expert.

TARGET ENTITY:
Name: GDPR
Type: Regulation

DETECTED ENTITIES IN CONTEXT:
- data protection [Legal Concept]
- European Union [Organization]

CONTEXT CHUNKS:
--- Chunk ID: test_001 ---
The GDPR regulates data protection in the European Union.
---

TASK:
Extract ALL relationships where "GDPR" is connected to the detected entities above.

RULES:
- Subject MUST be "GDPR" OR one of the detected entities
- Object MUST be one of the detected entities OR "GDPR"
- Use ONLY entities from the detected list above
- NO duplicate relations

OUTPUT FORMAT (JSON only, no other text):
{
  "relations": [
    {
      "subject": "entity_name",
      "predicate": "verb_phrase",
      "object": "entity_name",
      "chunk_ids": ["chunk_id"]
    }
  ]
}"""
    
    try:
        client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
        print(f"Prompt length: ~{len(prompt)//4} tokens")
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.0,
            stop=["```", "\n\nNote:", "Explanation:"]
        )
        
        print(f"✓ API call succeeded")
        print(f"  Finish reason: {response.choices[0].finish_reason}")
        
        content = response.choices[0].message.content
        print(f"  Content length: {len(content) if content else 0} chars")
        print(f"  Content is None: {content is None}")
        
        if content:
            print(f"\n  Raw content:")
            print(f"  {'-'*76}")
            print(f"  {content[:500]}")
            print(f"  {'-'*76}")
            
            # Try to parse JSON
            try:
                cleaned = content.strip().replace("```json", "").replace("```", "")
                parsed = json.loads(cleaned)
                print(f"\n  ✓ Parsed JSON successfully")
                print(f"  Relations found: {len(parsed.get('relations', []))}")
                if parsed.get('relations'):
                    print(f"  First relation: {parsed['relations'][0]}")
            except json.JSONDecodeError as e:
                print(f"\n  ✗ JSON parse failed: {e}")
                print(f"  Cleaned content (first 200 chars): {cleaned[:200]}")
        else:
            print(f"  ✗ Content is None or empty!")
        
        if hasattr(response, 'usage'):
            print(f"\n  Token usage:")
            print(f"    Prompt: {response.usage.prompt_tokens}")
            print(f"    Completion: {response.usage.completion_tokens}")
            print(f"    Total: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TOGETHER.AI API DEBUG TEST SUITE")
    print("="*80)
    print(f"Model: Qwen/Qwen2.5-7B-Instruct-Turbo")
    print(f"API Key present: {bool(os.getenv('TOGETHER_API_KEY'))}")
    
    results = []
    
    # Test 1: Basic connectivity
    results.append(("Basic Connectivity", test_basic_connectivity()))
    
    # Test 2: Simple JSON
    results.append(("Simple JSON", test_simple_json()))
    
    # Test 3: Relation extraction
    results.append(("Relation Extraction", test_minimal_relation_extraction()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - API is working correctly")
        print("  → Issue is likely in prompt format or entity list handling")
    else:
        print("✗ SOME TESTS FAILED - API has issues")
        print("  → Check API key, rate limits, or Together.ai status")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
