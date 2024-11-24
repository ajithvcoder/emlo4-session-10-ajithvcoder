import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import time
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Find the most recent metrics.csv file
csv_files = glob("logs/train/runs/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)
# TODO: Remove after debug
latest_csv = csv_files[-1]


### Confusion matrix 
## Train dataset
train_log_files = glob("logs/train/runs/*/train.log")
print(train_log_files)

def get_file_creation_time(filename):
    # Get the file creation time
    creation_time = os.path.getctime(filename)
    
    # Convert the creation time to a readable format
    readable_time = time.ctime(creation_time)
    
    return readable_time

for each in train_log_files:
    print(get_file_creation_time(each), each)

if not train_log_files:
    raise FileNotFoundError("No train.log file found")
latest_train_log_files = max(train_log_files, key=os.path.getctime)

## TODO: Remove after debug
latest_train_log_files = train_log_files[-1]

print(latest_train_log_files)
# exit()
base_path = latest_train_log_files.split("/train.log")[0]
shutil.copy(os.path.join(base_path,"train_confusion_matrix.png"), "train_confusion_matrix.png")

### Test dataset
eval_log_files = glob("logs/eval/runs/*/eval.log")
if not eval_log_files:
    raise FileNotFoundError("No eval.log file found")
latest_eval_log_files = max(eval_log_files, key=os.path.getctime)
## TODO: Remove after debug
latest_eval_log_files = eval_log_files[-1]
base_path = latest_eval_log_files.split("/eval.log")[0]
shutil.copy(os.path.join(base_path,"test_confusion_matrix.png"), "test_confusion_matrix.png")


# Read the CSV file
df = pd.read_csv(latest_csv)
print(df)
# Extract train_acc and val_acc for each epoch
train_acc = df.groupby('epoch')['train_acc_epoch'].last().reset_index()
val_acc = df.groupby('epoch')['val_acc'].last().reset_index()
train_loss = df.groupby('epoch')['train_loss_epoch'].last().reset_index()
val_loss = df.groupby('epoch')['val_loss'].last().reset_index()

# Merge the two dataframes on epoch
acc_df = pd.merge(train_acc, val_acc, on='epoch')

# Create training and validation accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(acc_df["epoch"] + 1, acc_df["train_acc_epoch"], label="Training Accuracy")
plt.plot(acc_df["epoch"] + 1, acc_df["val_acc"], label="Validation Accuracy")
print(acc_df)
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy over Steps")
plt.legend()
plt.savefig("train_val_acc.png")
plt.close()


# Merge the two dataframes on epoch
loss_df = pd.merge(train_loss, val_loss, on='epoch')

# Create training and validation accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(loss_df["epoch"] + 1, loss_df["train_loss_epoch"], label="Training Loss")
plt.plot(loss_df["epoch"] + 1, loss_df["val_loss"], label="Validation Loss")
print(loss_df)
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Training and Validation Loss over epochs")
plt.legend()
plt.savefig("train_val_loss.png")
plt.close()

