import os
import pandas as pd
import yaml
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt

def read_csv_to_dict(file_path):
    data_dict = {}
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for header in csv_reader.fieldnames:
            data_dict[header] = []

        for row in csv_reader:
            for header, value in row.items():
                if value:
                    data_dict[header].append(value)

    # last epoch is stored for testing run so taking last before one
    extract_csv_dict = {}
    extract_csv_dict['epoch'] = data_dict['epoch'][-2]
    extract_csv_dict['train_acc_epoch'] = data_dict['train_acc_epoch'][-1]
    extract_csv_dict['val_acc'] = data_dict['val_acc'][-1]
    extract_csv_dict['test_acc'] = data_dict['test_acc'][-1]

    extract_csv_dict['val_loss'] = data_dict['val_loss'][-1]
    extract_csv_dict['test_loss'] = data_dict['test_loss'][-1]
    
    return extract_csv_dict

def get_latest_timestamp(log_dir):
    timestamps = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    latest_timestamp = max(timestamps)
    return latest_timestamp

def get_metrics(log_dir, timestamp, mode):
    metrics = []
    metrics_dir = os.path.join(log_dir, mode, 'multiruns', timestamp)
    metrics_csv_dict = {}
    for run in os.listdir(metrics_dir):
        metrics_file = os.path.join(metrics_dir, run, "csv", "version_0", 'metrics.csv')
        if os.path.exists(metrics_file):
            metrics_csv_dict[int(run)] = read_csv_to_dict(metrics_file)
    return metrics_csv_dict


def get_hyperparams(log_dir, timestamp):
    hyperparams = {}
    hyperparams_dir = os.path.join(log_dir, 'train', 'multiruns', timestamp)
    for run in os.listdir(hyperparams_dir):
        hyperparams_file = os.path.join(hyperparams_dir, run, 'csv', 'version_0', 'hparams.yaml')
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, 'r') as file:
                params = yaml.safe_load(file)
            hyperparams[int(run)] = params

    # Extract patch_size, embed_dim along with key
    extract_hyperparams = {}
    for key, values in hyperparams.items():
        extract_hyperparams[int(key)] = {"base_model": values.get('base_model'), 
                                    "patch_size": values.get('patch_size'), "embed_dim": values.get('embed_dim')}
    return extract_hyperparams

def main_run():
    log_dir = 'logs'
    timestamp = get_latest_timestamp(os.path.join(log_dir, 'train/multiruns'))
    
    # Get train and test metrics
    train_metrics = get_metrics(log_dir, timestamp, 'train')
    # test_metrics = get_metrics(log_dir, timestamp, 'train')
    
    # Get hyperparams
    hyperparams = get_hyperparams(log_dir, timestamp)
    assert list(train_metrics.keys()).sort() == list(hyperparams.keys()).sort()

    data = [["Exp No"]]
    # print(train_metrics)
    metrics_keys = train_metrics[0].keys()
    hyperparams_keys = hyperparams[0].keys()
    data[0].extend(metrics_keys)
    data[0].extend(hyperparams_keys)
    exp_keys = list(train_metrics.keys())
    exp_keys.sort()
    for eachExp in exp_keys:
        exp_values = [eachExp]
        for each_param in metrics_keys:
            exp_values.append(train_metrics[eachExp][each_param])
        for each_param in hyperparams_keys:
            exp_values.append(hyperparams[eachExp][each_param])
        data.append(exp_values)

    # data_headers = train_metrics

    table = tabulate(data, headers="firstrow", tablefmt="pipe")
    
    # Display or save the merged data
    print("Train Metrics:\n", train_metrics)
    # print("Test Metrics:\n", test_metrics)
    print("Hyperparameters:\n", hyperparams)

    # Print the table to the console
    print(table)

    # Write the table to README.md
    with open("report.md", "w") as f:
        f.write("### Hyperparameters and test accuracy")
        f.write("\n")
        f.write("\n")
        f.write(table)

    # Extracting the experiment numbers, validation accuracies, and losses
    exp_nums = list(train_metrics.keys())
    exp_nums.sort()
    val_accs = [round(float(train_metrics[exp]['val_acc']), 2) for exp in exp_nums]
    val_losses = [round(float(train_metrics[exp]['val_loss']), 2) for exp in exp_nums]

    # Plotting the graph
    plt.figure(figsize=(10, 5))

    # Plotting validation accuracy
    plt.plot(exp_nums, val_accs, label='Validation Accuracy', marker='o', color='b')

    # Plotting validation loss
    plt.plot(exp_nums, val_losses, label='Validation Loss', marker='o', color='r')

    # Adding labels and title
    plt.xlabel('Experiment Number')
    plt.ylabel('Value')
    plt.title('Validation Accuracy and Loss Across Experiments')
    plt.legend()

    plt.savefig('validation_plot.png')

    # Write the table to README.md
    with open("report.md", "a") as f:
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("### Validation loss and validation accuracy plot")
        f.write("\n")
        f.write("![validation_plot](./validation_plot.png)")


    hyperparams_file = f"./logs/train/multiruns/{timestamp}/optimization_results.yaml"
    if os.path.exists(hyperparams_file):
        with open(hyperparams_file, 'r') as file:
            hparams_data = yaml.safe_load(file)

    exp_conf_file = f"./logs/train/multiruns/{timestamp}/0/.hydra/config.yaml"
    if os.path.exists(exp_conf_file):
        with open(exp_conf_file, 'r') as file:
            exp_data = yaml.safe_load(file)

    hparams_content = f"""
### Optuna Hyperparameter Optimization 
#### Experiment Name:  **{exp_data['experiment_name']}** 
#### Best Parameters: 
"""

    # Loop through the 'best_params' dictionary to add all the parameters in a generic way
    for param, value in hparams_data['best_params'].items():
        hparams_content += f"- **{param}**: {value}\n"

    # Add the best value (metric)
    hparams_content += f"""
## Best Value (Metric):
**{round(float(hparams_data['best_value']), 4)}**
"""

    # Write to README.md
    with open('report.md', 'a') as f:
        f.write("\n")
        f.write("\n")
        f.write(hparams_content)
    
    with open('best_model_checkpoint.txt', 'w') as f:
        f.write(f"./model_storage/epoch-checkpoint_patch_size-{hparams_data['best_params']['model.patch_size']}_embed_dim-{hparams_data['best_params']['model.embed_dim']}.ckpt")

    import shutil

    # Define the source file and destination folder
    source_file = 'best_model_checkpoint.txt'
    destination_folder = 'model_storage/'

    # Copy the file to the destination folder
    shutil.copy(source_file, destination_folder)

    print(f"{source_file} has been copied to {destination_folder}")


    # Define the path to the checkpoint file and folder containing .ckpt files
    checkpoint_file = 'best_model_checkpoint.txt'
    checkpoint_folder = 'model_storage'

    # Read the first line of the checkpoint file to get the file to keep
    with open(checkpoint_file, 'r') as f:
        keep_file = f.readline().strip()

    # Get the full path of the file to keep
    keep_file_path = os.path.join(checkpoint_folder, os.path.basename(keep_file))

    # Iterate over files in the checkpoint folder and delete unwanted .ckpt files
    for file in os.listdir(checkpoint_folder):
        file_path = os.path.join(checkpoint_folder, file)
        if file_path.endswith('.ckpt') and file_path != keep_file_path:
            os.remove(file_path)
            print(f"Removed: {file_path}")

    print(f"Kept: {keep_file_path}")



if __name__ == "__main__":
    main_run()
