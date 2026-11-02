"""Exercise the shipped checkpoint with user-created inputs, not a model stub.

These are functional serving tests. Generated images have no defect ground truth
and are not additional segmentation-accuracy evidence.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import io
import json
import threading
import zipfile

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

import semreview.app as service


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'REVIEWS', tmp_path)
    with TestClient(service.app) as client:
        yield client


def upload(client, size=(192, 160)):
    # Non-square, custom RGB input exercises the same contract as a user's image.
    y, x = np.indices((size[1], size[0]))
    image = np.stack([(x % 256), (y % 256), ((x + y) % 256)], axis=-1).astype('uint8')
    response = client.post('/api/inspect', files={'file': ('user-image.png', png(image), 'image/png')})
    assert response.status_code == 200, response.text
    return response.json()


def unpack(response):
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/zip'
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    return archive, json.loads(archive.read('report.json'))


def test_real_checkpoint_custom_image_review_lifecycle(client):
    health = client.get('/api/health').json()
    assert health['ready'] and len(health['model_id']) == 16
    record = upload(client)
    assert record['model_id'] == health['model_id']
    assert record['image_size_wh'] == [192, 160]
    assert record['inference_ms'] >= 0 and record['status'] == 'pending_review'
    key = record['review_id']
    original = client.get(f'/api/reviews/{key}/files/prediction.png').content
    original_mask = np.asarray(Image.open(io.BytesIO(original))) > 0
    assert original_mask.shape == (160, 192)
    assert not original_mask[:32].any() and not original_mask[:, :32].any()
    mask = np.zeros((160, 192), dtype='uint8')
    mask[50:60, 70:90] = 255
    mask[:8] = 255  # Excluded border edits cannot become measured area.
    response = client.post(f'/api/reviews/{key}/correct', files={'mask': ('edit.png', png(mask), 'image/png')})
    assert response.status_code == 200
    edited = response.json()
    assert edited['revision'] == 1 and edited['area_pixels'] == 200
    assert edited['status'] == 'corrected'
    accepted = client.post(f'/api/reviews/{key}/accept').json()
    assert accepted['status'] == 'accepted' and accepted['revision'] == 1
    archive, report = unpack(client.get(f'/api/reviews/{key}/export'))
    assert report['status'] == 'accepted' and report['area_pixels'] == 200
    assert archive.read('prediction.png') == original
    assert 'correction-1.png' in archive.namelist()
    assert (np.asarray(Image.open(io.BytesIO(archive.read('mask.png')))) > 0).sum() == 200
    rows = list(csv.DictReader(io.StringIO(archive.read('inspection.csv').decode())))
    assert rows[0]['status'] == 'accepted' and int(rows[0]['area_pixels']) == 200


@pytest.mark.parametrize('shape', [(64, 64), (128, 4097)])
def test_invalid_dimensions_do_not_create_review(client, shape):
    response = client.post('/api/inspect', files={'file': ('bad.png', png(np.zeros(shape, dtype='uint8')), 'image/png')})
    assert response.status_code == 400
    assert not list(service.REVIEWS.iterdir())


def test_invalid_bytes_and_oversized_upload_do_not_create_review(client):
    for body, expected in [(b'not an image', 400), (b'x' * (12 * 1024 * 1024 + 1), 413)]:
        response = client.post('/api/inspect', files={'file': ('bad.png', body, 'image/png')})
        assert response.status_code == expected
    assert not list(service.REVIEWS.iterdir())


def test_mismatched_correction_preserves_existing_revision(client):
    record = upload(client)
    key = record['review_id']
    before = client.get(f'/api/reviews/{key}/files/mask.png').content
    response = client.post(f'/api/reviews/{key}/correct', files={'mask': ('wrong.png', png(np.zeros((128, 128), dtype='uint8')), 'image/png')})
    assert response.status_code == 400
    assert client.get(f'/api/reviews/{key}').json()['revision'] == 0
    assert client.get(f'/api/reviews/{key}/files/mask.png').content == before
    assert client.get('/api/reviews/not-a-review').status_code == 404


def test_concurrent_user_uploads_have_separate_records(client):
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda _: upload(client), range(3)))
    assert len({r['review_id'] for r in records}) == 3
    for record in records:
        _, report = unpack(client.get(f"/api/reviews/{record['review_id']}/export"))
        assert report['review_id'] == record['review_id']


def test_export_waits_for_in_progress_revision(client, monkeypatch):
    record = upload(client)
    key = record['review_id']
    entered = threading.Event()
    original_iterdir = Path.iterdir

    def observed_iterdir(path):
        if path == service.REVIEWS / key:
            entered.set()
        return original_iterdir(path)

    monkeypatch.setattr(Path, 'iterdir', observed_iterdir)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with service.LOCK:
            future = pool.submit(client.get, f'/api/reviews/{key}/export')
            # Export must not inspect files until the revision writer releases.
            assert not entered.wait(.15)
        archive, report = unpack(future.result(timeout=5))
    assert entered.is_set() and report['review_id'] == key
    assert 'mask.png' in archive.namelist()


def test_report_read_waits_for_revision_commit(client, monkeypatch):
    record = upload(client)
    key = record['review_id']
    report_path = service.REVIEWS / key / 'report.json'
    entered = threading.Event()
    original_read = Path.read_text

    def observed_read(path, *args, **kwargs):
        if path == report_path:
            entered.set()
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', observed_read)
    committed = json.loads(original_read(report_path))
    committed.update(status='accepted', revision=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with service.LOCK:
            # Simulate the interval between truncating and completing a write.
            report_path.write_text('{"status":')
            future = pool.submit(client.get, f'/api/reviews/{key}')
            assert not entered.wait(.15)
            report_path.write_text(json.dumps(committed))
        response = future.result(timeout=5)
    assert entered.is_set() and response.status_code == 200
    assert response.json()['status'] == 'accepted'
    assert response.json()['revision'] == 1
