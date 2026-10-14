"""Unit tests for multi-variant detection in ocr_server.py.

These run WITHOUT torch: the detector is faked, so the geometry (rotation
mappings, dedup) and the adaptive control flow can be verified locally for free
before anything is rented. Run:

    python deploy/test_detection.py

Everything here exists because a single detection pass left 32-74 of 1034
delivered pages with untranslated text (see llm_handoff/14_thrashing_postmortem.md
- verify in isolation before spending money).
"""
import os
import sys
import types

import numpy as np

# --- stub fastapi so ocr_server imports without the real package ------------
_fake = types.ModuleType("fastapi")


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def get(self, *a, **k):
        return self

    def post(self, *a, **k):
        return self


class _HTTPExc(Exception):
    def __init__(self, status_code=None, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


_fake.FastAPI = _Dummy
_fake.File = lambda *a, **k: None
_fake.Header = lambda *a, **k: None
_fake.UploadFile = object
_fake.HTTPException = _HTTPExc
sys.modules.setdefault("fastapi", _fake)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_server  # noqa: E402

TOL = 1
FAILURES = []


def check(label, ok):
    print("  %-42s %s" % (label, "OK" if ok else "FAIL"))
    if not ok:
        FAILURES.append(label)


def close(a, b):
    return all(abs(int(x) - int(y)) <= TOL for x, y in zip(a, b))


def marker_box(arr):
    ys, xs = np.where(arr == 255)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def test_mapping():
    print("rotation mappings (against numpy ground truth):")
    base = np.zeros((12, 20), dtype=np.uint8)
    base[3:7, 5:9] = 255
    b90 = marker_box(np.rot90(base, 1))
    b270 = marker_box(np.rot90(base, 3))
    check("rot90 box maps back", close(ocr_server._rot90_box_to_base(b90, 20), [5, 3, 8, 6]))
    check("rot270 box maps back", close(ocr_server._rot270_box_to_base(b270, 12), [5, 3, 8, 6]))
    check("rot90 mask round-trip", np.array_equal(np.rot90(np.rot90(base, 1), -1), base))
    check("rot270 mask round-trip", np.array_equal(np.rot90(np.rot90(base, 3), 1), base))


def test_dedup():
    print("dedup:")
    boxes = [[10, 10, 50, 30], [11, 11, 51, 31], [200, 200, 260, 240], [200, 200, 260, 240]]
    check("2 duplicate pairs collapse to 2", len(ocr_server._dedup_boxes(boxes)) == 2)
    check("zero-area boxes dropped", len(ocr_server._dedup_boxes([[5, 5, 5, 5]])) == 0)


class _Block:
    def __init__(self, box):
        x0, y0, x1, y1 = box
        self.lines = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]


class _FakeDetector:
    """Returns `n` boxes regardless of the input, and counts its calls."""

    def __init__(self, n):
        self.n, self.calls = n, 0

    def __call__(self, arr):
        self.calls += 1
        h, w = arr.shape[:2]
        blocks = []
        for i in range(self.n):
            y = 10 + i * 12
            if y + 8 >= h:
                break
            blocks.append(_Block([10, y, min(w - 10, 120), y + 8]))
        mask = np.zeros((h, w), dtype=np.uint8)
        for b in blocks:
            mask[10:120, 10:120] = 255
        return mask, mask, blocks


def test_adaptive():
    from PIL import Image
    print("adaptive flow (fake detector):")
    det = _FakeDetector(20)
    ocr_server.get_detector = lambda: det
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    boxes, _mask = ocr_server.detect(img)
    check("plenty of text -> single pass", det.calls == 1)
    check("plenty of text -> boxes returned", len(boxes) == 20)

    det = _FakeDetector(1)
    ocr_server.get_detector = lambda: det
    boxes, _mask = ocr_server.detect(img)
    check("little text -> escalates to all passes",
          det.calls == len(ocr_server.DET_VARIANTS))
    check("escalation adds boxes", len(boxes) > 1)
    check("boxes stay inside the image",
          all(0 <= b[0] < b[2] <= 400 and 0 <= b[1] < b[3] <= 600 for b in boxes))


def main():
    test_mapping()
    test_dedup()
    test_adaptive()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        return 1
    print("ALL DETECTION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
