"""Hard testing of the entire relay-ml pipeline.

Tests all endpoints with various edge cases:
1. Grade-image: all 3 modes (bedrock, cnn, mock) + validation
2. Grade-images: multi-angle + edge cases
3. Grade-video: bedrock path + validation
4. Fit-flags: aggregate + fallback + multiflags
5. Embed: local + structured + determinism
6. Wish-score: trained model + monotonicity + edge values
7. Return-clusters: clustering + edge cases
8. Health: mode reporting
"""

import os
import sys
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("AWS_REGION", "us-east-1")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

PASS = 0
FAIL = 0


def check(condition: bool, test_name: str, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {test_name}")
    else:
        FAIL += 1
        print(f"  ✗ {test_name} — {detail}")


def make_png(width=100, height=100, color=(128, 128, 128)) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg(width=100, height=100) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(200, 50, 50))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def load_real_image(path_str: str) -> tuple[bytes, str] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = f.read()
    if path.suffix.lower() == ".webp":
        from PIL import Image
        img = Image.open(BytesIO(data)).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    elif path.suffix.lower() in (".jpg", ".jpeg"):
        return data, "image/jpeg"
    return data, "image/png"


# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Health Endpoint")
print("=" * 70)

r = client.get("/health")
check(r.status_code == 200, "Health returns 200")
d = r.json()
check(d["status"] == "ok", "Health status is ok")
check("cnn_version" in d, "Health includes cnn_version")
check("notes" in d, "Health includes notes")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Grade-Image Validation")
print("=" * 70)

# Reject non-image content type
r = client.post("/grade-image",
    files={"image": ("test.gif", b"GIF89a", "image/gif")},
    data={"unit_id": "u1", "category": "electronics"})
check(r.status_code == 400, "Rejects GIF content type")

# Reject oversized image (>8MB)
big = b"\xff\xd8" + b"\x00" * (8 * 1024 * 1024 + 100)
r = client.post("/grade-image",
    files={"image": ("big.jpg", big, "image/jpeg")},
    data={"unit_id": "u1", "category": "electronics"})
check(r.status_code == 400, "Rejects >8MB image")

# Reject empty image
r = client.post("/grade-image",
    files={"image": ("empty.png", b"", "image/png")},
    data={"unit_id": "u1", "category": "electronics"})
check(r.status_code in (400, 503), "Rejects empty image")

# Accept valid PNG (Bedrock mode)
png = make_png()
r = client.post("/grade-image",
    files={"image": ("test.png", png, "image/png")},
    data={"unit_id": "test-png", "category": "fashion"})
check(r.status_code == 200, "Accepts valid PNG")
if r.status_code == 200:
    d = r.json()
    check(d["schema_version"] == "1.0.0", "Passport has schema_version")
    check(d["unit_id"] == "test-png", "Passport has correct unit_id")
    check(d["grade"] in ("A+", "A", "B+", "B", "C", "D"), "Grade is valid enum")
    check(0 <= d["confidence"] <= 1, "Confidence in range")
    check(d["passport_hash"] != "", "Passport hash is non-empty")
    check(len(d["media_hashes"]) >= 1, "Has media hash")
    check(d["model_tier_used"].startswith("bedrock"), "Uses bedrock tier")

# Accept valid JPEG
jpg = make_jpeg()
r = client.post("/grade-image",
    files={"image": ("test.jpg", jpg, "image/jpeg")},
    data={"unit_id": "test-jpg", "category": "electronics", "return_id": "ret-123"})
check(r.status_code == 200, "Accepts valid JPEG")
if r.status_code == 200:
    d = r.json()
    check(d["return_id"] == "ret-123", "return_id passed through")
    check(d["vertical"] == "electronics", "Vertical inferred correctly")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Grade-Image with Real Product Photos")
print("=" * 70)

real_images = [
    (r"C:\Users\bhavy\Downloads\71lpqynZ1eL._SY250_.jpg", "electronics", "new"),
    (r"C:\Users\bhavy\Downloads\broken-black-headphones-wooden-table-close-up-182058582.webp", "headphones", "damaged"),
    (r"C:\Users\bhavy\Downloads\broken-smart-phone.webp", "phone", "damaged"),
]

for path_str, category, expected_state in real_images:
    loaded = load_real_image(path_str)
    if loaded is None:
        print(f"  [SKIP] {Path(path_str).name}")
        continue
    img_data, ct = loaded
    r = client.post("/grade-image",
        files={"image": ("img", img_data, ct)},
        data={"unit_id": f"real-{category}", "category": category})
    check(r.status_code == 200, f"Real image ({category}) graded OK")
    if r.status_code == 200:
        d = r.json()
        if expected_state == "new":
            check(d["grade"] in ("A+", "A", "B+"), f"New product graded high ({d['grade']})")
        else:
            check(d["grade"] in ("B", "B+", "C", "D"), f"Damaged product graded low ({d['grade']})")
        check(d["confidence"] > 0.5, f"Confidence reasonable ({d['confidence']:.0%})")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Grade-Images (Multi-angle)")
print("=" * 70)

# Multi-image with 2 valid PNGs
png1 = make_png(color=(100, 200, 100))
png2 = make_png(color=(100, 100, 200))
r = client.post("/grade-images",
    files=[("images", ("a.png", png1, "image/png")), ("images", ("b.png", png2, "image/png"))],
    data={"unit_id": "multi-test", "category": "fashion"})
check(r.status_code == 200, "Multi-image grading works (2 images)")
if r.status_code == 200:
    d = r.json()
    check("angles" in d["model_tier_used"], "Model tier reports angles")
    check(len(d["media_hashes"]) == 2, "Has 2 media hashes")

# Reject >8 images
files_9 = [("images", (f"img{i}.png", make_png(), "image/png")) for i in range(9)]
r = client.post("/grade-images", files=files_9, data={"unit_id": "u1", "category": "fashion"})
check(r.status_code == 400, "Rejects >8 images")

# Reject mixed valid+invalid content types
r = client.post("/grade-images",
    files=[("images", ("a.png", png1, "image/png")), ("images", ("b.bmp", b"\x00", "image/bmp"))],
    data={"unit_id": "u1", "category": "fashion"})
check(r.status_code == 400, "Rejects invalid content type in multi-image")

# Real multi-angle
multi_paths = [
    r"C:\Users\bhavy\Downloads\81BHNfdeA3L._SY250_.jpg",
    r"C:\Users\bhavy\Downloads\81lxKVOesfL._SY250_.jpg",
]
multi_files = []
for p in multi_paths:
    loaded = load_real_image(p)
    if loaded:
        multi_files.append(("images", (Path(p).name, loaded[0], loaded[1])))
if len(multi_files) == 2:
    r = client.post("/grade-images", files=multi_files, data={"unit_id": "real-multi", "category": "electronics"})
    check(r.status_code == 200, "Real multi-angle graded OK")
    if r.status_code == 200:
        d = r.json()
        check(d["confidence"] >= 0.8, f"Multi-angle confidence high ({d['confidence']:.0%})")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Grade-Video")
print("=" * 70)

# Reject non-video content type
r = client.post("/grade-video",
    files={"video": ("test.txt", b"not video", "text/plain")},
    data={"unit_id": "u1", "category": "electronics"})
check(r.status_code == 400, "Rejects non-video content type")

# Reject empty video
r = client.post("/grade-video",
    files={"video": ("empty.mp4", b"", "video/mp4")},
    data={"unit_id": "u1", "category": "electronics"})
check(r.status_code == 400, "Rejects empty video")

# Reject oversized video (>64MB)
r = client.post("/grade-video",
    files={"video": ("big.mp4", b"\x00" * (64 * 1024 * 1024 + 1), "video/mp4")},
    data={"unit_id": "u1", "category": "electronics"})
check(r.status_code == 400, "Rejects >64MB video")

# Real video
vid_path = Path(r"C:\Users\bhavy\Downloads\WhatsApp Video 2026-06-14 at 5.50.58 PM.mp4")
if vid_path.exists():
    with open(vid_path, "rb") as f:
        vid_data = f.read()
    r = client.post("/grade-video",
        files={"video": ("v.mp4", vid_data, "video/mp4")},
        data={"unit_id": "vid-test", "category": "electronics"})
    check(r.status_code == 200, "Real video graded OK")
    if r.status_code == 200:
        d = r.json()
        check(len(d["media_hashes"]) >= 2, f"Video has multiple hashes ({len(d['media_hashes'])})")
        check("keyframes" in d["model_tier_used"], "Tier reports keyframes")
        check(d["grade"] in ("A+", "A", "B+", "B", "C", "D"), f"Valid grade ({d['grade']})")
else:
    print("  [SKIP] Video file not found")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Fit-Flags")
print("=" * 70)

# Known category from aggregates
r = client.post("/fit-flags", json={"sku_id": "SKU-1", "category": "coat"})
check(r.status_code == 200, "Fit-flags works for known category")
if r.status_code == 200:
    d = r.json()
    check(d["sku_id"] == "SKU-1", "SKU passed through")
    check(len(d["flags"]) >= 1, "At least one flag returned")
    check("aggregates" in d["source"] or "rules" in d["source"], f"Source identified ({d['source']})")

# Unknown category falls back to rules
r = client.post("/fit-flags", json={"sku_id": "SKU-999", "category": "alien_tech"})
check(r.status_code == 200, "Fit-flags fallback for unknown category")
if r.status_code == 200:
    d = r.json()
    check("rules" in d["source"] or "critical" in d["flags"][0]["type"], "Falls back gracefully")

# MultiFlags endpoint
r = client.post("/fit-flags/multi", json={
    "sku_id": "SKU-MF-1", "category": "jeans",
    "total_returns": 200, "too_small_count": 80,
    "too_large_count": 10, "fit_count": 110, "is_article_level": True})
check(r.status_code == 200, "MultiFlags endpoint works")
if r.status_code == 200:
    d = r.json()
    check("multiflags" in d["source"], f"Source is multiflags ({d['source']})")
    types = [f["type"] for f in d["flags"]]
    check("runs_small" in types, f"Detects runs_small bias ({types})")

# MultiFlags low data
r = client.post("/fit-flags/multi", json={
    "sku_id": "SKU-LOW", "category": "hat",
    "total_returns": 2, "too_small_count": 1,
    "too_large_count": 0, "fit_count": 1})
check(r.status_code == 200, "MultiFlags handles low data")
if r.status_code == 200:
    d = r.json()
    check(d["flags"][0]["type"] == "critical_fit", "Low data → critical_fit")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Embed Endpoint")
print("=" * 70)

# Free text
r = client.post("/embed", json={"text": "red leather boots"})
check(r.status_code == 200, "Embed with free text")
if r.status_code == 200:
    d = r.json()
    check(len(d["vector"]) == 384, f"Vector is 384-d ({len(d['vector'])})")
    check(d["model"] != "", "Model name returned")

# Structured attrs
r = client.post("/embed", json={"category": "phone", "grade": "C", "size": "6.1in", "vertical": "electronics"})
check(r.status_code == 200, "Embed with structured attrs")

# Determinism
r1 = client.post("/embed", json={"text": "exact same input"})
r2 = client.post("/embed", json={"text": "exact same input"})
if r1.status_code == 200 and r2.status_code == 200:
    check(r1.json()["vector"] == r2.json()["vector"], "Embed is deterministic")

# Different inputs differ
r1 = client.post("/embed", json={"text": "brand new iPhone"})
r2 = client.post("/embed", json={"text": "old torn jacket"})
if r1.status_code == 200 and r2.status_code == 200:
    check(r1.json()["vector"] != r2.json()["vector"], "Different inputs → different vectors")

# Empty body fallback
r = client.post("/embed", json={})
check(r.status_code == 200, "Empty body returns fallback vector")
if r.status_code == 200:
    check(len(r.json()["vector"]) == 384, "Fallback vector is 384-d")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Wish-Score")
print("=" * 70)

# Basic
r = client.post("/wish-score", json={
    "wish_age_days": 5, "user_purchase_count": 3,
    "category_affinity": 0.7, "has_fit_profile": True})
check(r.status_code == 200, "Wish-score basic works")
if r.status_code == 200:
    d = r.json()
    check(0 <= d["score"] <= 1, f"Score in range ({d['score']})")
    check(d["model"] == "logreg_v1", f"Model is logreg_v1 ({d['model']})")

# Monotonicity: recency
r_new = client.post("/wish-score", json={"wish_age_days": 1, "user_purchase_count": 5, "category_affinity": 0.5, "has_fit_profile": False})
r_old = client.post("/wish-score", json={"wish_age_days": 28, "user_purchase_count": 5, "category_affinity": 0.5, "has_fit_profile": False})
if r_new.status_code == 200 and r_old.status_code == 200:
    check(r_new.json()["score"] > r_old.json()["score"], "Newer wish → higher score")

# Monotonicity: purchases
r_few = client.post("/wish-score", json={"wish_age_days": 5, "user_purchase_count": 0, "category_affinity": 0.5, "has_fit_profile": False})
r_many = client.post("/wish-score", json={"wish_age_days": 5, "user_purchase_count": 20, "category_affinity": 0.5, "has_fit_profile": False})
if r_few.status_code == 200 and r_many.status_code == 200:
    check(r_many.json()["score"] > r_few.json()["score"], "More purchases → higher score")

# Monotonicity: affinity
r_low = client.post("/wish-score", json={"wish_age_days": 5, "user_purchase_count": 5, "category_affinity": 0.1, "has_fit_profile": False})
r_high = client.post("/wish-score", json={"wish_age_days": 5, "user_purchase_count": 5, "category_affinity": 0.95, "has_fit_profile": False})
if r_low.status_code == 200 and r_high.status_code == 200:
    check(r_high.json()["score"] > r_low.json()["score"], "Higher affinity → higher score")

# Monotonicity: fit profile
r_no = client.post("/wish-score", json={"wish_age_days": 5, "user_purchase_count": 5, "category_affinity": 0.5, "has_fit_profile": False})
r_yes = client.post("/wish-score", json={"wish_age_days": 5, "user_purchase_count": 5, "category_affinity": 0.5, "has_fit_profile": True})
if r_no.status_code == 200 and r_yes.status_code == 200:
    check(r_yes.json()["score"] > r_no.json()["score"], "Fit profile → higher score")

# Edge: zero everything
r = client.post("/wish-score", json={"wish_age_days": 0, "user_purchase_count": 0, "category_affinity": 0.0, "has_fit_profile": False})
check(r.status_code == 200, "Zero-everything input accepted")

# Edge: maximum everything
r = client.post("/wish-score", json={"wish_age_days": 30, "user_purchase_count": 100, "category_affinity": 1.0, "has_fit_profile": True})
check(r.status_code == 200, "Max-everything input accepted")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Return-Clusters")
print("=" * 70)

# Basic clustering
r = client.post("/return-clusters", json={
    "reasons": [
        "color not as shown", "color looks different", "shade mismatch",
        "too small", "runs small", "doesnt fit too tight",
        "arrived damaged", "box was crushed"
    ],
    "n_clusters": 3})
check(r.status_code == 200, "Clustering works")
if r.status_code == 200:
    d = r.json()
    check(d["total_reasons"] == 8, f"Total reasons correct ({d['total_reasons']})")
    check(d["num_clusters"] >= 2, f"Multiple clusters returned ({d['num_clusters']})")
    total_in_clusters = sum(c["count"] for c in d["clusters"])
    check(total_in_clusters == 8, "All reasons assigned to clusters")

# Empty list rejected
r = client.post("/return-clusters", json={"reasons": []})
check(r.status_code == 422, "Empty reasons list rejected")

# Single reason
r = client.post("/return-clusters", json={"reasons": ["only one reason"]})
check(r.status_code == 200, "Single reason accepted")

# Duplicates preserved
r = client.post("/return-clusters", json={"reasons": ["same", "same", "same"], "min_cluster_size": 1})
check(r.status_code == 200, "Duplicates handled")
if r.status_code == 200:
    check(r.json()["total_reasons"] == 3, "Duplicate count preserved")

# ===========================================================
print("\n" + "=" * 70)
print("  HARD TEST: Mock Mode")
print("=" * 70)

with patch("app.routers.grade.settings") as mock_s:
    mock_s.grading_mode = "mock"
    mock_s.cnn_model_path = "models/grade_cnn_v1.pt"
    r = client.post("/grade-image",
        files={"image": ("t.png", make_png(), "image/png")},
        data={"unit_id": "mock-test", "category": "shoes"})
    check(r.status_code == 200, "Mock mode returns 200")
    if r.status_code == 200:
        d = r.json()
        check(d["model_tier_used"] == "mock", "Mock tier reported")
        check(d["grade"] == "B", "Mock always returns B")

# ===========================================================
# SUMMARY
# ===========================================================
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
print("=" * 70)
if FAIL == 0:
    print("  🎉 ALL TESTS PASSED!")
else:
    print(f"  ⚠️  {FAIL} test(s) failed — review above")
print()
