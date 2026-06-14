"""Quick script to see the raw Bedrock response for debugging."""
import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

with open(r"C:\Users\bhavy\Downloads\61DapK5FZrL._SY250_.jpg", "rb") as f:
    img_bytes = f.read()

client = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt = """You are an AI product condition grader for a circular commerce platform.

Analyze this product image (category: electronics) and assess its physical condition.

Return a JSON object with these fields:
- "grade": one of "A+", "A", "B+", "B", "C", "D"
  - A+ = pristine/sealed, looks brand new, no visible wear
  - A = like new, minimal signs of use
  - B+ = good condition, light cosmetic wear only
  - B = fair, visible wear but fully functional
  - C = poor, significant damage or defects
  - D = heavily damaged, may not be functional
- "defect_type": one of "none", "scuff", "crack", "stain", "tear", "dent", "discoloration", "missing_part", "screen_damage", "water_damage", "functional_fault", "other"
  - Use "none" if the product appears undamaged (grade A+ or A)
- "confidence": float 0.0-1.0 how confident you are in your assessment
- "description": brief description of the product condition (1-2 sentences)

Important: If the product looks new or undamaged, grade it A+ or A with defect_type "none".
Only report defects you can actually see in the image.

Return ONLY valid JSON, no other text."""

response = client.converse(
    modelId="amazon.nova-lite-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {"image": {"format": "jpeg", "source": {"bytes": img_bytes}}},
                {"text": prompt},
            ],
        }
    ],
    inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
)

raw = response["output"]["message"]["content"][0]["text"]
print("RAW BEDROCK RESPONSE:")
print(raw)
print()
try:
    parsed = json.loads(raw)
    print("PARSED:")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError:
    print("(Could not parse as JSON)")
