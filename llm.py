import os
from google import genai
from google.genai import types

# Create client ONCE
client = genai.Client(
    api_key="AIzaSyARxEtJrfIFhldxXW204HT_ykSRZTdtDUI" #place your gemini API key here"
)

MODEL_NAME = "gemini-3-flash-preview"  # Change the model based on cost

def guess_tamil_word(letters):
    """
    letters: list[str]  → ['க','வ']
    returns: str        → 'வணக்கம்'
    """

    text = "".join(letters)

    prompt = f"""
You are a Tamil language expert.

Input letters (possibly incomplete, noisy):
{text}

Task:
- Guess the closest meaningful Tamil word
- Use common spoken Tamil
- Return ONLY the word
- No explanation
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]

    config = types.GenerateContentConfig(
        max_output_tokens=10,          # VERY IMPORTANT
        temperature=0.2,               # stable
        top_p=0.9
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )

    # Handle case where response.text might be None
    if response and hasattr(response, 'text') and response.text:
        return response.text.strip()
    else:
        # Return the original text if API fails
        return text
