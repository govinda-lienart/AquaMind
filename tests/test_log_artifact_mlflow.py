from scripts.log_artifact_mlflow import count_images, load_results_csv


def test_count_images(tmp_path):
    (tmp_path / 'frame_001.jpg').write_text('x')
    (tmp_path / 'frame_002.jpg').write_text('x')
    (tmp_path / 'frame_003.png').write_text('x')
    (tmp_path / 'labels.txt').write_text('x')
    assert count_images(str(tmp_path)) == 3

def test_load_results_csv(tmp_path):
    csv_path = tmp_path / 'results.csv'
    csv_path.write_text(
        " epoch, train/box_loss, train/cls_loss, train/dfl_loss,"
        " val/box_loss, val/cls_loss, val/dfl_loss,"
        " metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),"
        " lr/pg0, lr/pg1, lr/pg2\n"
        "0, 1.0, 0.5, 0.3, 0.9, 0.4, 0.2, 0.8, 0.7, 0.6, 0.4, 0.01, 0.01, 0.01\n"
    )
    df = load_results_csv(str(tmp_path))
    assert 'epoch' in df.columns
    assert 'train/box_loss' in df.columns
    assert len(df) == 1
