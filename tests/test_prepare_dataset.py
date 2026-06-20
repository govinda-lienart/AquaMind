import os
import shutil
import yaml
from scripts.prepare_dataset import split_train_val, create_dataset_dirs, write_dataset_card, write_yolo_yaml, main


def test_split_train_val():
    frames = [(i, f'frame_{i}.png') for i in range(10)]
    train, val = split_train_val(frames)
    assert len(train) == 8
    assert len(val)   == 2
    assert len(train) + len(val) == len(frames)

def test_create_dataset_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_dataset_dirs('test_ds')
    assert os.path.isdir('dataset/test_ds/images/train')
    assert os.path.isdir('dataset/test_ds/images/val')
    assert os.path.isdir('dataset/test_ds/labels/train')
    assert os.path.isdir('dataset/test_ds/labels/val')

def test_write_dataset_card(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs('dataset/test_ds')
    write_dataset_card('test_ds', [1, 2], [], n_train=8, n_val=2, git_commit='abc1234')
    with open('dataset/test_ds/dataset_card.yaml') as f:
        card = yaml.safe_load(f)
    assert card['dataset_name']       == 'test_ds'
    assert card['annotation_set_ids'] == [1, 2]
    assert card['num_train']          == 8
    assert card['num_val']            == 2

def test_write_yolo_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_yolo_yaml('test_ds')
    with open('dataset.yaml') as f:
        data = yaml.safe_load(f)
    assert data['nc']    == 2
    assert data['names'] == ['danio_rerio', 'reflection']
    assert 'kpt_shape'   not in data

def test_main(db_conn, tmp_path, monkeypatch):
    frame_dir = tmp_path / 'frames' / 'frames_IMG_0350_20260101_2000'
    frame_dir.mkdir(parents=True)
    (frame_dir / 'frame_360_IMG_0350.png').write_bytes(b'')
    shutil.copy('config.yaml', tmp_path)
    monkeypatch.chdir(tmp_path)
    main(db_conn)
