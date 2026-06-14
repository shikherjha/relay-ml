"""Quick live test for /grade-and-price."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

with open(r"C:\Users\bhavy\Downloads\71lpqynZ1eL._SY250_.jpg", "rb") as f:
    img = f.read()

r = client.post(
    "/grade-and-price",
    files=[("images", ("p.jpg", img, "image/jpeg"))],
    data={"unit_id": "test-gap", "category": "electronics", "original_price": "4999", "age_days": "60"},
)
d = r.json()
print(f"Status: {r.status_code}")
print(f"Grade: {d['grade']} | Resale grade: {d['resale_grade']}")
print(f"Price range: INR {d['price_range']['min']:.0f} - {d['price_range']['max']:.0f}")
mean = (d["price_range"]["min"] + d["price_range"]["max"]) / 2
print(f"List price (mean): INR {mean:.0f}")
print(f"Rationale: {d['pricing_rationale']}")
print(f"Confidence: {d['confidence']:.0%}")
print(f"Defects: {len(d['defects'])}")
