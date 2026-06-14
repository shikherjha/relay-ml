"""Quick test: video grading via Bedrock path."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

vid_path = Path(r"C:\Users\bhavy\Downloads\WhatsApp Video 2026-06-14 at 5.50.58 PM.mp4")
if not vid_path.exists():
    print("Video not found!")
    sys.exit(1)

with open(vid_path, "rb") as f:
    vid_data = f.read()

print(f"Video: {vid_path.name} ({len(vid_data)/1024/1024:.1f} MB)")
r = client.post(
    "/grade-video",
    files={"video": ("v.mp4", vid_data, "video/mp4")},
    data={"unit_id": "test-video-bedrock", "category": "electronics"},
)

d = r.json()
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Grade: {d['grade']} (confidence: {d['confidence']:.0%})")
    if d["defects"]:
        for defect in d["defects"]:
            print(f"Defect: {defect['type']} ({defect['severity']})")
            if defect.get("description"):
                print(f"  → {defect['description'][:120]}")
    else:
        print("No damage detected!")
    print(f"Tier: {d['model_tier_used']}")
    print(f"Media hashes: {len(d['media_hashes'])}")
else:
    print(f"Error: {d.get('detail', d)}")
