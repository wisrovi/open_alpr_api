"""Unit tests for VehicleTracker and temporal plate caching."""

import pytest

from app.tracker import VehicleTracker, compute_iou


def test_compute_iou_identical_boxes() -> None:
    """Validate that identical bounding boxes have IoU of 1.0."""
    box = [10, 10, 50, 50]
    assert compute_iou(box, box) == 1.0


def test_compute_iou_disjoint_boxes() -> None:
    """Validate that disjoint non-overlapping bounding boxes have IoU of 0.0."""
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 30, 30]
    assert compute_iou(box_a, box_b) == 0.0


def test_tracker_assigns_and_tracks_ids() -> None:
    """Validate that tracker assigns track IDs and tracks vehicle across consecutive frames."""
    tracker = VehicleTracker(iou_threshold=0.4, max_age=5)

    # Frame 1: Detection of vehicle at [100, 100, 200, 200]
    frame1 = [{"bbox": [100, 100, 200, 200], "conf": 0.9, "class": 2}]
    res1 = tracker.update(frame1)
    assert len(res1) == 1
    det1, track_id1, plate1, conf1 = res1[0]
    assert track_id1 == 1
    assert plate1 is None

    # Frame 2: Slightly moved vehicle [102, 101, 202, 201] matches track 1
    frame2 = [{"bbox": [102, 101, 202, 201], "conf": 0.89, "class": 2}]
    res2 = tracker.update(frame2)
    assert len(res2) == 1
    det2, track_id2, plate2, conf2 = res2[0]
    assert track_id2 == 1


def test_tracker_caches_and_returns_plate() -> None:
    """Validate that recorded high-confidence plates are cached and returned on subsequent frames."""
    tracker = VehicleTracker(iou_threshold=0.4, min_cached_conf=85.0)

    # Frame 1: Vehicle detected
    frame1 = [{"bbox": [50, 50, 150, 150]}]
    res1 = tracker.update(frame1)
    track_id = res1[0][1]

    # Record recognized plate for this track
    tracker.record_plate(track_id, "2000DFL", 91.5)

    # Frame 2: Next frame for same vehicle returns cached plate without calling OCR
    frame2 = [{"bbox": [52, 51, 151, 150]}]
    res2 = tracker.update(frame2)
    assert len(res2) == 1
    _, returned_tid, cached_plate, cached_conf = res2[0]
    assert returned_tid == track_id
    assert cached_plate == "2000DFL"
    assert cached_conf == 91.5


def test_tracker_prunes_stale_tracks() -> None:
    """Validate that stale tracks beyond max_age frames are dropped."""
    tracker = VehicleTracker(max_age=2)
    tracker.update([{"bbox": [10, 10, 30, 30]}])
    assert len(tracker.tracks) == 1

    # Update with empty frames exceeding max_age
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.tracks) == 0
