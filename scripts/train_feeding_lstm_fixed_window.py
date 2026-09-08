"""
- Loads the per-window embedding sequences (train_embeddings_*.pt / test_embeddings_*.pt) built by build_feeding_embeddings_fixed_window.py — each window is a (45, 384) tensor + label
- Wraps them in a Dataset / DataLoader
- Defines a small LSTM + linear classifier head (the only part that trains — the DINOv2 backbone was frozen and already run in the embeddings step)
- Trains with early stopping: keep the checkpoint at min test loss, not the last epoch (the fix that caught the 200-epoch overfitting — see diary.md, Stage 7 Part 2)
- Reports accuracy / precision / recall on the test split

usage: python -m scripts.train_feeding_lstm_fixed_window
"""
import os
import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
from scripts.console import banner, banner_sub
from scripts.video_utils import grab_video_name
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn


VIDEO_RUN_NAME = 'IMG_2349_appearance_2026_08_12_1926'
BACKBONE_NAME = 'dinov2_vits14'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# STEP 1 — load the per-window embedding sequences (.pt)
banner("STEP 1 — load train / test embedding sequences")
parquet_path, *_ = grab_video_name(VIDEO_RUN_NAME)
emb_dir = os.path.join(os.path.dirname(parquet_path), "feeding_train_test", "embeddings")
train_path = os.path.join(emb_dir, f"train_embeddings_{BACKBONE_NAME}.pt")
test_path  = os.path.join(emb_dir, f"test_embeddings_{BACKBONE_NAME}.pt")
train_windows = torch.load(train_path, weights_only=False)  # weights_only=False -> full pickle, needed because this is my own list-of-dicts, not a flat state_dict (safe loader only handles tensors/state_dicts)
test_windows  = torch.load(test_path,  weights_only=False)
logger.info("")
logger.info(f"train: {len(train_windows)} windows | test: {len(test_windows)} windows")
logger.info(f"window keys: {list(train_windows[0].keys())}")
logger.info(f"first embeddings shape: {tuple(train_windows[0]['embeddings'].shape)}")

# STEP 2 — Dataset + DataLoader (serve one window: (45, 384) tensor + label)
banner("STEP 2 — Dataset / DataLoader") # DataLoader's job is the looping, shuffling, and batching,
class FeedingWindowDataset(Dataset):
  """Defines a new dataset class that inherits PyTorch's Dataset interface, so a DataLoader can batch and shuffle it."""
  def __init__(self, windows):
      """Stores the incoming list of windows onto the object so the other methods can access it later."""
      self.windows = windows
  def __len__(self):
      """Tells the DataLoader how many samples total exist in the dataset."""
      return len(self.windows)
  def __getitem__(self, idx): # idx the Sampler (hidden operation) inside the DataLoader  generates the index values, and the DataLoader is what calls your dataset with them.
      """Fetches and formats one training sample by index."""
      w = self.windows[idx] # Grabs the one dict at position idx out of the stored list, so the rest of the method can pull its embeddings and label from it.
      emb = w["embeddings"]
      label = torch.tensor(w["label"], dtype=torch.long) # Converts the plain Python int label into a long-type tenso(int64), the format CrossEntropyLoss requires it for comparison against the model's output.
      return emb, label 
train_ds = FeedingWindowDataset(train_windows)
test_ds = FeedingWindowDataset(test_windows) 
logger.info(f"train_ds: {len(train_ds)} samples | test_ds: {len(test_ds)} samples")
sample_emb, sample_label = train_ds[0]
logger.info(f"sample emb: {tuple(sample_emb.shape)}, label: {sample_label}")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True) # every epoch, shuffles all 136 indices into a new random order, then hands them out to __getitem__ 8 at a time to build each batch.
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False) # Since nothing is learned during eval no shuffling needed for evaluation/same order every run, easier to debu  — fixed order, no gradient updates, just forward passes to check performance.
logger.info(f"train_loader: {len(train_loader)} batches | test_loader: {len(test_loader)} batches") #136 windows ÷ 8 = 17  /// 59 windows ÷ 8 = 7.375 
 
# STEP 3 — define the LSTM + classifier head (nn.Module) # build the structure — the blueprint of layers and how they connect
banner("STEP 3 — model")
class FeedingLSTMClassifier(nn.Module): 
    def __init__(self, input_size=384, hidden_size=64,num_classes=2):
                                # hidden_size=64 is the size of the vector the LSTM produces at its last time step —> that's what feeds into the head
                                # num_classes (2) describes the head's output side — the final answer size you want.
        super().__init__() # calls the parent class (nn.Module's) constructor and exectures its hidden set up -  it builds a canbitet to srtore all the layers of my model and its parameters 
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) #  builds and stores the LSTM  layer. 
        self.head = nn.Linear(hidden_size, num_classes) 
    def forward(self, x):
        # Every i run the model - dorward executes fresh on batch tensor
        # so for 17 train batches - forward runs 17 times - processing 8 windowns thorhg LSTM/head - producing 9 sets of class scoress 
        output, (h_n, c_n) = self.lstm(x)
            # output fore each of the 45 steps 
            # h_n final hidden state - >>> LSTM final memory (64 lenght summary vecot) after reading thw whole sequence
            # h_n consists of (num_layers, batch, hidden_size) (1, batch, 64)
        """
            h_n = [                          # level 1: layers (size 1 — you have 1 LSTM layer)
                [                             # level 2: batch (size 8 — one per window in the batch)
                    [0.12, -0.5, ..., 0.03],  # level 3: the 64 numbers = the summary vector for window 1
                    [0.44, 0.01, ..., -0.2],  # the 64 numbers for window 2
                    ... (8 of these total)
                ]
            ]
        """
            # c_n final cell state (internal - no neededd)
        last_hidden = h_n[-1] # (1, batch, 64) take the last of the list
        # the [-1] literally drops from (1, 8, 64) → (8, 64) (no range:used -) the folder wrapepr dissapears - using -1 removes the outer layer 
        logits = self.head(last_hidden)
        return logits    
         # e.g [-1.2, 3.4] — raw scores - a higher number in position 1 means the model currently favors "feeding strike" for this window.
    
 # STEP 4 — one training epoch + one eval pass (helper functions)
banner("STEP 4 — train / eval helpers")
# your coklo           

def train_one_epoch(model, loader, criterion, optimizer):
    """One training epoch: forward, loss, backward, optimizer step per batch -> returns mean train loss"""
    model.train() # Sets the model to training modert # "it flips a mode flag that Dropout and BatchNorm layers check  # but here model just 2 layers so no effect
                # BatchNorm: it keeps the numbers flowing through the network from getting too big or too small or too erratic, like a thermostat that keeps a room's temperature 
                # Dropout -  it randomly "blinds" part of the network on purpose, so it can't get lazy and rely on just a few shortcuts — like practicing a sport with one arm tied behind your back sometimes, so you develop all your skills, not just your favorite move.

    total_loss = 0.0 # "PyTorch only computes the loss for one batch at a time — it's on me to add those up loss up at for ex at epoch-level averag
    for embeddings, labels in loader: # see step 2 : train and test loader
        embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)# the model was alrady moved to cpu....we also need to move the embeddings and the labels
        logits = model(embeddings) # model(x) call the method which calls model forwad() from step 4
        loss = criterion(logits, labels) # runss crossentroypy on the batchs logits(raw class scoeres) against the labels( produicing one numebr - how wrong the model was on this batch)
        optimizer.zero_grad() #  clears leftover gradients from the previous batch 
        loss.backward() # runs backpropagation - computes how much each weight in the mdoel contributed to this batch error.
        optimizer.step() # updates the weights, nudging each one a little in the direction that reduces the loss, sized you learning rate (0.001)
        total_loss += loss.item() # Adds this batch's loss value onto the running total, so by the end of the epoch I have the sum of all batches' losses
    return total_loss / len(loader) # that is the nyumebr of batches (eg 17 for train). divifing this gives the mean loss per batch for the epoch....number to compare epoch to epoch when looking fro overtfiitng  

def eval_one_epoch(model, loader, criterion):
    """measures how the model performs on the test set - never changes the weights"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():# tells pythoch to no track computational graph - never also calls backward() - not learning just measuring + "with" makes sure that no grad function is stopped once finished and turn on again back on grad
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
            logits = model(embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# STEP 5 — training loop with early stopping (keep min-test-loss checkpoint)
banner("STEP 5 — training loop")

model = FeedingLSTMClassifier().to(DEVICE) # argument function step 5 -  calls __init__, which builds and stores the lstm and head layers  defined in STEP 3 
criterion = nn.CrossEntropyLoss() # is what the train/eval_one_epoch expects as argument
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer - update model's weight during optimizer.step() #  it is a cabinet the cabinet is really storing "the pieces that make up the network cabinet is really storing "the pieces that make up the network,super().__init__() built an empty cabinet. Then self.lstm = nn.LSTM(...) and self.head = nn.Linear(...) each got automatically filed into it. model.parameters() is just "give me everything that's in the cabinet.it's tracking every individual trainable number inside both of them, ready to hand to the optimizer.
num_epochs = 100 # how many passes through the full training set
best_test_loss = float("inf") # starts as infinity so the very first epoch's test loss automatically counts as an improvement
checkpoint_path = "best_feeding_lstm.pt" # the filename where you'll save the best-performing model's weights

for epoch in range(num_epochs): # Inside this loop, each iteration is one full pass over the training data
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss = eval_one_epoch(model, test_loader, criterion)
    logger.info(f"epoch {epoch+1}/{num_epochs} | train_loss  {train_loss:.4f} | test_loss {test_loss:.4f}")  # epoch+1 — since range() starts at 0, this prints "epoch 1" instead of "epoch 0" for the first pass, more human-readable  # :.4f — formats each loss to 4 decimal places, so the printed numbers stay readable instead of showing 15 digits

    if test_loss < best_test_loss: # targetting smalles value...
        best_test_loss = test_loss
        torch.save(model.state_dict(), checkpoint_path) # state_dict() saves just the learned numbers (the weights), not the model's blueprint/code.
        logger.info(f"  -> new best test_loss {best_test_loss:.4f}, checkpoint saved")

#  STEP 6 — load the best checkpoint back and report final metrics.
banner("STEP 6 — final evaluation")

best_model = FeedingLSTMClassifier().to(DEVICE) # This calls __init__ again, so I get a brand-new LSTM + head with random weights, sitting on the right device.
best_model.load_state_dict(torch.load(checkpoint_path)) # .load_state_dict(...) copies each of those tensors into the matching layer of best_model, overwriting the random init.
best_model.eval() # Sets the mode flag to eval. Here it barely matters for 2 layer model

all_preds = [] # what the model guessed (0 or 1 per window)
all_labels = [] # he ground truth from  my labeling.

with torch.no_grad():
    for embeddings, labels in test_loader:
        embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
        logits = best_model(embeddings)
        preds = logits.argmax(dim=1) # (batch, 2) raw scores -> (batch,) predicted class index (the larger score wins)
        all_preds.append(preds.cpu()) # all_preds is a list of ~8 small tensors
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds) # torch.cat ("concatenate") joins the list of ~8 batch tensors end-to-end into one tensor of ~59 numbers (the whole test set)
all_labels = torch.cat(all_labels)

y_true = all_labels.numpy() # sklearn wants plain NumPy arrays, not torch tensors
y_pred = all_preds.numpy()

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label=1) # class 1 = feeding strike
rec  = recall_score(y_true, y_pred, pos_label=1)

logger.info("")
logger.info(f"test accuracy  {acc:.3f}")
logger.info(f"test precision {prec:.3f}  (of predicted strikes, how many were real)")
logger.info(f"test recall    {rec:.3f}  (of real strikes, how many we caught)")
