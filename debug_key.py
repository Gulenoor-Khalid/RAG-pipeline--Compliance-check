#!/usr/bin/env python3
"""
Debug API key format and test different approaches
"""

def analyze_key(api_key):
    """Analyze the API key format"""
    print(f"Key length: {len(api_key)}")
    print(f"Starts with 'gsk_': {api_key.startswith('gsk_')}")
    print(f"Key preview: {api_key[:12]}...{api_key[-8:]}")
    print(f"Contains whitespace: {' ' in api_key or '\n' in api_key or '\t' in api_key}")
    
    # Check for common issues
    if len(api_key) < 50:
        print("⚠️  WARNING: Key seems too short")
    
    if not api_key.startswith('gsk_'):
        print("⚠️  WARNING: Key doesn't start with 'gsk_'")
    
    # Clean the key
    clean_key = api_key.strip()
    if clean_key != api_key:
        print("🔧 Key had whitespace - cleaning...")
        return clean_key
    
    return api_key

def test_with_different_models(api_key):
    """Test with different Groq models"""
    from groq import Groq
    
    models_to_try = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile", 
        "mixtral-8x7b-32768"
    ]
    
    client = Groq(api_key=api_key)
    
    for model in models_to_try:
        try:
            print(f"Trying model: {model}")
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            print(f"✅ SUCCESS with {model}")
            return True
        except Exception as e:
            print(f"❌ Failed with {model}: {str(e)}")
    
    return False

if __name__ == "__main__":
    api_key = input("Paste your complete Groq API key: ").strip()
    
    print("=== API KEY ANALYSIS ===")
    clean_key = analyze_key(api_key)
    
    print("\n=== TESTING API ===")
    test_with_different_models(clean_key)