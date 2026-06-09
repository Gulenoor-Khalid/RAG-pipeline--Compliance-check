#!/usr/bin/env python3
"""
Quick test script to verify Groq API key works
"""

def test_groq_key(api_key):
    """Test if Groq API key is working"""
    try:
        from groq import Groq
        
        print(f"Testing API key: {api_key[:8]}...")
        
        client = Groq(api_key=api_key)
        
        # Simple test query
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast model
            messages=[
                {"role": "user", "content": "Hello! Just testing the API connection."}
            ],
            max_tokens=50,
            temperature=0.1,
        )
        
        response = completion.choices[0].message.content
        print("✅ SUCCESS! API key is working.")
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    # Replace this with your actual API key
    api_key = input("Enter your Groq API key: ").strip()
    
    if not api_key:
        print("No API key provided!")
    else:
        test_groq_key(api_key)