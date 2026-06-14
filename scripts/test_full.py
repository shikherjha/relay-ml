"""Full testing script for relay-ml endpoints.

Run with: .venv\Scripts\python.exe scripts/test_full.py

Tests:
1. Single image grading (5-6 different products)
2. Multi-image grading (3-4 images of same product)
3. Video grading (2-3 videos)

Usage:
  Place your test files in a folder and update the paths below.
  Or run interactively and provide paths when prompted.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Ensure AWS region is set
os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def print_passport(data: dict, label: str = "") -> None:
    """Pretty-print a ConditionPassport response."""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  Grade:       {data['grade']} (numeric: {data['grade_numeric']:.2f})")
    print(f"  Confidence:  {data['confidence']:.2%}")
    print(f"  Defects:     {len(data['defects'])} found")
    for d in data['defects']:
        print(f"    - {d['type']} ({d['severity']}) conf={d.get('confidence', 'N/A')}")
        if d.get('description'):
            print(f"      {d['description']}")
    if not data['defects']:
        print(f"    No damage detected!")
    print(f"  Disposition: {data['disposition_hint']}")
    print(f"  Packaging:   {data['packaging_state']}")
    print(f"  Vertical:    {data['vertical']}")
    print(f"  Category:    {data['category']}")
    print(f"  Model tier:  {data['model_tier_used']}")
    print(f"  Media hash:  {data['media_hashes'][0][:20]}..." if data['media_hashes'] else "  No media hash")
    print(f"  Passport OK: {bool(data['passport_hash'])}")
    print(f"{'='*60}\n")


def detect_content_type(path: Path) -> str:
    """Detect content type from file extension."""
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".mp4":
        return "video/mp4"
    elif ext == ".mov":
        return "video/quicktime"
    elif ext == ".webm":
        return "video/webm"
    elif ext == ".avi":
        return "video/x-msvideo"
    return "application/octet-stream"


# ============================================================
# TEST 1: Single image grading on different products
# ============================================================

def test_single_images(image_paths: list[str], categories: list[str]) -> None:
    """Grade individual product images one at a time."""
    print("\n" + "="*60)
    print("  TEST 1: Single Image Grading")
    print("="*60)

    for i, (img_path, category) in enumerate(zip(image_paths, categories), 1):
        path = Path(img_path)
        if not path.exists():
            print(f"\n  [SKIP] File not found: {img_path}")
            continue

        content_type = detect_content_type(path)
        with open(path, "rb") as f:
            img_bytes = f.read()

        response = client.post(
            "/grade-image",
            files={"image": (path.name, img_bytes, content_type)},
            data={"unit_id": f"test-single-{i}", "category": category},
        )

        if response.status_code == 200:
            print_passport(response.json(), f"Image {i}: {path.name} (category: {category})")
        else:
            print(f"\n  [ERROR] Image {i}: {path.name} → {response.status_code}")
            print(f"  Detail: {response.json().get('detail', response.text)}")


# ============================================================
# TEST 2: Multi-image grading (same product, multiple angles)
# ============================================================

def test_multi_images(image_paths: list[str], category: str) -> None:
    """Grade multiple images of the same product in one call."""
    print("\n" + "="*60)
    print("  TEST 2: Multi-Image Grading (same product, multiple angles)")
    print("="*60)

    files = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            print(f"  [SKIP] File not found: {img_path}")
            continue
        content_type = detect_content_type(path)
        with open(path, "rb") as f:
            files.append(("images", (path.name, f.read(), content_type)))

    if not files:
        print("  No valid image files found!")
        return

    print(f"  Sending {len(files)} images for category: {category}")

    response = client.post(
        "/grade-images",
        files=files,
        data={"unit_id": "test-multi-1", "category": category},
    )

    if response.status_code == 200:
        print_passport(response.json(), f"Multi-angle ({len(files)} images, category: {category})")
    else:
        print(f"\n  [ERROR] → {response.status_code}")
        print(f"  Detail: {response.json().get('detail', response.text)}")


# ============================================================
# TEST 3: Video grading
# ============================================================

def test_videos(video_paths: list[str], categories: list[str]) -> None:
    """Grade product videos via keyframe extraction."""
    print("\n" + "="*60)
    print("  TEST 3: Video Grading")
    print("="*60)

    for i, (vid_path, category) in enumerate(zip(video_paths, categories), 1):
        path = Path(vid_path)
        if not path.exists():
            print(f"\n  [SKIP] File not found: {vid_path}")
            continue

        content_type = detect_content_type(path)
        with open(path, "rb") as f:
            vid_bytes = f.read()

        print(f"\n  Grading video {i}: {path.name} ({len(vid_bytes) / 1024 / 1024:.1f} MB)")

        response = client.post(
            "/grade-video",
            files={"video": (path.name, vid_bytes, content_type)},
            data={"unit_id": f"test-video-{i}", "category": category},
        )

        if response.status_code == 200:
            print_passport(response.json(), f"Video {i}: {path.name} (category: {category})")
        else:
            print(f"\n  [ERROR] Video {i}: {path.name} → {response.status_code}")
            print(f"  Detail: {response.json().get('detail', response.text)}")


# ============================================================
# MAIN — fill in your test file paths below
# ============================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  RELAY ML — Full Endpoint Testing")
    print("#"*60)

    # -------------------------------------------------------
    # TEST 1: Single images — different products
    # Put your image paths and categories here:
    # -------------------------------------------------------
    single_images = [
        r"C:\Users\bhavy\Downloads\61DapK5FZrL._SY250_.jpg",
        # Add more paths below:
        # r"C:\path\to\damaged_phone.jpg",
        # r"C:\path\to\new_shoes.png",
        # r"C:\path\to\used_laptop.jpg",
        # r"C:\path\to\torn_shirt.jpg",
        # r"C:\path\to\headphones.jpg",
    ]

    single_categories = [
        "electronics",
        # "phone",
        # "shoes",
        # "laptop",
        # "fashion",
        # "headphones",
    ]

    if single_images:
        test_single_images(single_images, single_categories)

    # -------------------------------------------------------
    # TEST 2: Multi-image — same product from different angles
    # Put 3-4 images of the same product here:
    # -------------------------------------------------------
    multi_images = [
        # r"C:\path\to\product_front.jpg",
        # r"C:\path\to\product_back.jpg",
        # r"C:\path\to\product_side.jpg",
        # r"C:\path\to\product_bottom.jpg",
    ]
    multi_category = "electronics"

    if multi_images:
        test_multi_images(multi_images, multi_category)
    else:
        print("\n  [SKIP] No multi-image paths configured. Add paths to the script.")

    # -------------------------------------------------------
    # TEST 3: Video grading
    # Put video file paths here:
    # -------------------------------------------------------
    videos = [
        # r"C:\path\to\product_video_1.mp4",
        # r"C:\path\to\product_video_2.mp4",
        # r"C:\path\to\product_video_3.mov",
    ]
    video_categories = [
        # "electronics",
        # "fashion",
        # "shoes",
    ]

    if videos:
        test_videos(videos, video_categories)
    else:
        print("\n  [SKIP] No video paths configured. Add paths to the script.")

    # -------------------------------------------------------
    # BONUS: Test embed + wish-score
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("  BONUS: Embed + Wish-Score")
    print("="*60)

    r = client.post("/embed", json={"text": "red leather shoes size 9"})
    if r.status_code == 200:
        d = r.json()
        print(f"  /embed: dim={len(d['vector'])}, model={d['model']}")
    else:
        print(f"  /embed: ERROR {r.status_code} — {r.json().get('detail', r.text)}")

    r = client.post("/wish-score", json={
        "wish_age_days": 2, "user_purchase_count": 8,
        "category_affinity": 0.9, "has_fit_profile": True
    })
    if r.status_code == 200:
        d = r.json()
        print(f"  /wish-score: score={d['score']}, model={d['model']}")
    else:
        print(f"  /wish-score: ERROR {r.status_code}")

    print("\n" + "#"*60)
    print("#  DONE")
    print("#"*60 + "\n")
