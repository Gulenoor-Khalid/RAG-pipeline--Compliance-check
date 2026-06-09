#!/usr/bin/env python3
"""
Test with exact same configuration as the Streamlit app
"""

def test_app_config(api_key):
    """Test with the exact same setup as app.py"""
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Use exact same parameters as in llm.py
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # Same model as app
            messages=[
                {
                    "role": "system", 
                    "content": "You are a construction compliance expert. Analyze building codes and project specifications to provide accurate, technical answers."
                },
                {
                    "role": "user", 
                    "content": "What is the minimum fire exit width?"
                }
            ],
            max_tokens=150,
            temperature=0.1,  # Same temperature as app
        )
        
        response_text = completion.choices[0].message.content
        print("✅ SUCCESS! The API key works with app configuration.")
        print(f"Response: {response_text}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR with app config: {str(e)}")
        
        # Try with a simpler model
        try:
            print("\n🔄 Trying with simpler model...")
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=50,
                temperature=0.1,
            )
            response = completion.choices[0].message.content
            print("✅ SUCCESS with simpler model!")
            print(f"Response: {response}")
            print("\n💡 Try using 'llama-3.1-8b-instant' model in the app instead")
            return True
        except Exception as e2:
            print(f"❌ Also failed with simpler model: {str(e2)}")
        
        return False

if __name__ == "__main__":
    api_key = input("Enter your Groq API key: ").strip()
    test_app_config(api_key)