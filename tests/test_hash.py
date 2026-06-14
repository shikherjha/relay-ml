from app.utils.hash import canonical_json_hash, passport_hash, sha256_hex


def test_sha256_hex_hashes_bytes() -> None:
    assert (
        sha256_hex(b"test")
        == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    )


def test_canonical_json_hash_is_key_order_independent() -> None:
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}

    assert canonical_json_hash(left) == canonical_json_hash(right)


def test_passport_hash_excludes_existing_passport_hash() -> None:
    body = {"unit_id": "unit-1", "passport_hash": "old"}
    assert passport_hash(body) == canonical_json_hash({"unit_id": "unit-1"})
