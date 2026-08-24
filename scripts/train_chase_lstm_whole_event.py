"""usage:  python -m scripts.train_chase_lstm_whole_event
Phase F (whole-event variant) - LSTM fed one full variable-length sequence per labelled event
(the entire attack burst through any following/path-tracing tail) - the "textbook" way to use
an LSTM's native variable-length handling. Parked here (2026-08-19) in favor of the windowed
variant, which matches deployment's fixed-size sliding-window scanner more closely and produced
the more trustworthy result - see diary.md, Stage 7, for the full comparison."""


# IMPORTS

import os
from datetime import datetime
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')  # avoids popup windows of produced plots
import matplotlib.pyplot as plt
from scripts.console import banner, banner_sub
from scripts.chasing_features import grab_video_name, trim_to_calibration, build_pairs, build_sequences
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# CONSTANT
LABELS_XLS_PATH = 'output_fish_tracker/chase_labels.xlsx'
VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
RANDOM_SEED = 42
FEATURE_COLS = ['distance_cm_smooth_w15', 'max_speed_either', 'min_alignment_either_deg']  # same 3 signals the pruned sklearn baseline settled on, fed raw per-frame instead of mean/min/max
HIDDEN_SIZE = 16  # small on purpose - only ~30 training events, a bigger LSTM would just overfit
NUM_EPOCHS = 60
LEARNING_RATE = 0.005

torch.manual_seed(RANDOM_SEED)


# MODEL

class ChaseLSTM(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, padded, lengths):
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden_last, _) = self.lstm(packed)  # hidden_last: [1, batch, hidden_size] - the LSTM's own summary of the whole sequence
        return self.classifier(hidden_last.squeeze(0)).squeeze(-1)  # -> [batch] raw logits (no sigmoid - BCEWithLogitsLoss applies it)


# MAIN FUNCTION

def main():
    banner('STEP 1 - LOAD chase_labels.xlsx')
    labels = pd.read_excel(LABELS_XLS_PATH)
    logger.info(f'{labels.shape[0]} labeled events ({(labels["label"]==1).sum()} positive, {(labels["label"]==0).sum()} negative)')

    banner('STEP 2 - LOAD TRACKS + BUILD PAIRWISE FEATURES')
    parquet_path, pixels_per_cm, calibration_secs, surface_y_px, bottom_y_px, frame_number_end = grab_video_name(VIDEO_RUN_NAME)
    df = pd.read_parquet(parquet_path)
    df = trim_to_calibration(df, calibration_secs, frame_number_end)
    pairs = build_pairs(df, pixels_per_cm)
    pairs['min_alignment_either_deg'] = pairs['min_alignment_either_deg'].fillna(180)  # "no burst = no aim to report" - worst-case value
    logger.info(pairs[['frame_number', 'fish_id_a', 'fish_id_b', 'distance_cm', 'closing_speed_cm_s']].head().to_string())

    banner('STEP 3 - SLICE WHOLE VARIABLE-LENGTH EVENTS')
    all_sequences = build_sequences(pairs, labels, 'whole_event', RANDOM_SEED)
    n_pos = sum(1 for s in all_sequences if s['label'] == 1)
    n_neg = sum(1 for s in all_sequences if s['label'] == 0)
    logger.info(f'{len(all_sequences)} total events ({n_pos} positive, {n_neg} negative)')

    banner('STEP 4 - REUSE PHASE D/E TRAIN/TEST SPLIT (same event_ids)')
    split_folder = os.path.join(os.path.dirname(parquet_path), 'chase_train_test')
    train_event_ids = set(pd.read_parquet(os.path.join(split_folder, 'train_df.parquet'))['event_id'].unique())
    test_event_ids = set(pd.read_parquet(os.path.join(split_folder, 'test_df.parquet'))['event_id'].unique())
    train_sequences = [s for s in all_sequences if s['event_id'] in train_event_ids]
    test_sequences = [s for s in all_sequences if s['event_id'] in test_event_ids]
    logger.info(f'{len(train_sequences)} train events ({sum(s["label"] for s in train_sequences)} positive), '
                f'{len(test_sequences)} test events ({sum(s["label"] for s in test_sequences)} positive)')

    banner('STEP 5 - NORMALIZE FEATURES (fit on train only)')
    train_frames = pd.concat([s['sequence_df'][FEATURE_COLS] for s in train_sequences], ignore_index=True)
    feature_mean = train_frames.mean()
    feature_std = train_frames.std()
    logger.info(f'feature_mean:\n{feature_mean}\nfeature_std:\n{feature_std}')

    def to_tensor(sequence_df):
        normed = (sequence_df[FEATURE_COLS] - feature_mean) / feature_std
        return torch.tensor(normed.values, dtype=torch.float32)

    banner('STEP 6 - BUILD PADDED TENSORS')

    def build_batch(sequences):
        tensors = [to_tensor(s['sequence_df']) for s in sequences]
        lengths = torch.tensor([len(t) for t in tensors], dtype=torch.int64)
        labels_tensor = torch.tensor([s['label'] for s in sequences], dtype=torch.float32)
        padded = pad_sequence(tensors, batch_first=True)  # [batch, max_len, num_features]
        return padded, lengths, labels_tensor

    X_train, lengths_train, y_train = build_batch(train_sequences)
    X_test, lengths_test, y_test = build_batch(test_sequences)
    logger.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

    banner('STEP 7 - DEFINE MODEL')
    model = ChaseLSTM(num_features=len(FEATURE_COLS), hidden_size=HIDDEN_SIZE)
    logger.info(model)

    banner('STEP 8 - TRAIN')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train, lengths_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            logger.info(f'epoch {epoch:3d}  loss {loss.item():.4f}')

    banner('STEP 9 - EVALUATE ON TEST EVENTS')
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test, lengths_test)
        test_pred = (torch.sigmoid(test_logits) >= 0.5).int().numpy()

    logger.info(f'\n{classification_report(y_test.numpy(), test_pred, target_names=["not_chase", "chase"])}')

    stamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    output_folder = os.path.join(split_folder, 'output_train_chase_lstm_whole_event', stamp)
    os.makedirs(output_folder, exist_ok=True)

    cm = confusion_matrix(y_test.numpy(), test_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=['not_chase', 'chase']).plot(ax=ax)
    ax.set_title('lstm_whole_event')
    path_plot = os.path.join(output_folder, 'lstm_whole_event_confusion_matrix.png')
    fig.savefig(path_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'saved confusion matrix -> {path_plot}')

    banner_sub("SAVE RUN CONFIG")
    run_config = {
        'video_run_name': VIDEO_RUN_NAME, 'random_seed': RANDOM_SEED, 'feature_cols': FEATURE_COLS,
        'hidden_size': HIDDEN_SIZE, 'num_epochs': NUM_EPOCHS, 'learning_rate': LEARNING_RATE,
        'n_train_events': len(train_sequences), 'n_test_events': len(test_sequences),
    }
    with open(os.path.join(output_folder, 'run_config.yaml'), 'w') as f:
        yaml.dump(run_config, f, sort_keys=False)
    logger.info(f'saved run_config -> {output_folder}/run_config.yaml')


# ENTRY POINT

if __name__ == "__main__":
    main()
