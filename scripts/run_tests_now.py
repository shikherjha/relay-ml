"""Run live tests on the user's actual files."""
import os
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def load_image(path_str: str) -> tuple[bytes, str]:
    """Load image, convert webp to PNG if needed."""
    path = Path(path_str)
    with open(path, "rb") as f:
        data = f.read()

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return data, "image/jpeg"
    elif ext == ".png":
        return data, "image/png"
    elif ext == ".webp":
        from PIL import Image
        img = Image.open(BytesIO(data)).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    return data, "image/jpeg"


def print_result(data: dict, label: str) -> None:
    """Print grading result."""
    print(f"\n  [{label}]")
    print(f"  Grade:       {data['grade']} (numeric: {data['grade_numeric']:.2f})")
    print(f"  Confidence:  {data['confidence']:.0%}")
    defects = data["defects"]
    if defects:
        for d in defects:
            desc = d.get("description", "")
            print(f"  Defect:      {d['type']} ({d['severity']}) — {desc[:80]}")
    else:
        print(f"  Defects:     None — product appears undamaged")
    print(f"  Disposition: {data['disposition_hint']}")
    print(f"  Packaging:   {data['packaging_state']}")
    print(f"  Model tier:  {data['model_tier_used']}")


# ===========================================================
# TEST 1: Single image grading — 3 different products
# ===========================================================
print("\n" + "=" * 60)
print("  TEST 1: Single Image Grading (3 different products)")
print("=" * 60)

test1_files = [
    (r"C:\Users\bhavy\Downloads\71lpqynZ1eL._SY250_.jpg", "electronics", "Product 1: Electronics item"),
    (r"C:\Users\bhavy\Downloads\broken-black-headphones-wooden-table-close-up-182058582.webp", "headphones", "Product 2: Broken headphones"),
    (r"C:\Users\bhavy\Downloads\broken-smart-phone.webp", "phone", "Product 3: Broken smartphone"),
]

for path_str, category, label in test1_files:
    if not Path(path_str).exists():
        print(f"\n  [SKIP] {path_str} not found")
        continue

    img_data, ct = load_image(path_str)
    r = client.post(
        "/grade-image",
        files={"image": ("img", img_data, ct)},
        data={"unit_id": f"test-{category}", "category": category},
    )
    if r.status_code == 200:
        print_result(r.json(), label)
    else:
        print(f"\n  [ERROR] {label}: {r.status_code} — {r.json().get('detail', r.text)}")


# ===========================================================
# TEST 2: Multi-image grading — same product, 2 angles
# ===========================================================
print("\n\n" + "=" * 60)
print("  TEST 2: Multi-Image Grading (same product, 2 angles)")
print("=" * 60)

multi_files = [
    r"C:\Users\bhavy\Downloads\81BHNfdeA3L._SY250_.jpg",
    r"C:\Users\bhavy\Downloads\81lxKVOesfL._SY250_.jpg",
]

files_payload = []
for p in multi_files:
    if not Path(p).exists():
        print(f"  [SKIP] {p} not found")
        continue
    img_data, ct = load_image(p)
    files_payload.append(("images", (Path(p).name, img_data, ct)))

if files_payload:
    print(f"  Sending {len(files_payload)} images...")
    r = client.post(
        "/grade-images",
        files=files_payload,
        data={"unit_id": "test-multi-angle", "category": "electronics"},
    )
    if r.status_code == 200:
        print_result(r.json(), f"Multi-angle ({len(files_payload)} images)")
    else:
        print(f"\n  [ERROR] {r.status_code} — {r.json().get('detail', r.text)}")


# ===========================================================
# TEST 3: Video grading
# ===========================================================
print("\n\n" + "=" * 60)
print("  TEST 3: Video Grading")
print("=" * 60)

video_path = r"C:\Users\bhavy\Downloads\WhatsApp Video 2026-06-14 at 5.50.58 PM.mp4"

if Path(video_path).exists():
    with open(video_path, "rb") as f:
        vid_data = f.read()
    print(f"  Video size: {len(vid_data) / 1024 / 1024:.1f} MB")

    r = client.post(
        "/grade-video",
        files={"video": ("video.mp4", vid_data, "video/mp4")},
        data={"unit_id": "test-video-1", "category": "electronics"},
    )
    if r.status_code == 200:
        data = r.json()
        print_result(data, "Video grading")
        print(f"  Media hashes: {len(data['media_hashes'])} (video + keyframe hashes)")
    else:
        print(f"\n  [ERROR] {r.status_code} — {r.json().get('detail', r.text)}")
else:
    print(f"  [SKIP] Video not found: {video_path}")


# ===========================================================
# SUMMARY
# ===========================================================
print("\n\n" + "=" * 60)
print("  BONUS: Embed + Wish-Score")
print("=" * 60)

r = client.post("/embed", json={"text": "broken smartphone cracked screen"})
if r.status_code == 200:
    d = r.json()
    print(f"  /embed: dim={len(d['vector'])}, model={d['model']}")

r = client.post("/wish-score", json={
    "wish_age_days": 2, "user_purchase_count": 8,
    "category_affinity": 0.9, "has_fit_profile": True
})
if r.status_code == 200:
    d = r.json()
    print(f"  /wish-score: score={d['score']}, model={d['model']}")

print("\n  DONE!\n")
