import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time
import random 

class StormCNET1(nn.Module):
    def __init__(self, input_channels):
        super(StormCNET1, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 2
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 3
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Dropout(0.1),
        )
        
        # Calculate the output size of CNN layers
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 241, 321)
            sample_output = self.cnn_layers(sample_input)
            self.flat_features = sample_output.view(1, -1).size(1)
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.SiLU(),
            nn.Linear(64, 4),  # 4 classes: cyclogenesis, full typhoon, cyclolysis, no cyclone
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



def load_typhoon_data(file_path):
    print("Loading typhoon data...")
    typhoon_data = pd.read_csv(file_path)
    typhoon_data['Cyclogenesis Start'] = pd.to_datetime(typhoon_data['Cyclogenesis Start'])
    typhoon_data['Cyclogenesis End'] = pd.to_datetime(typhoon_data['Cyclogenesis End'])
    typhoon_data['Typhoon Start'] = pd.to_datetime(typhoon_data['Typhoon Start'])
    typhoon_data['Typhoon End'] = pd.to_datetime(typhoon_data['Typhoon End'])
    typhoon_data['Cyclolysis Start'] = pd.to_datetime(typhoon_data['Cyclolysis Start'])
    typhoon_data['Cyclolysis End'] = pd.to_datetime(typhoon_data['Cyclolysis End'])
    print("Typhoon data loaded.")
    return typhoon_data

def get_true_label(date, typhoon_data):
    for _, row in typhoon_data.iterrows():
        if row['Cyclogenesis Start'] <= date <= row['Cyclogenesis End']:
            return 0  # cyclogenesis
        elif row['Typhoon Start'] <= date <= row['Typhoon End']:
            return 1  # full typhoon
        elif row['Cyclolysis Start'] <= date <= row['Cyclolysis End']:
            return 2  # cyclolysis
    return 3  # no cyclone

def evaluate_model(model, ds, typhoon_data, device, num_samples=5000):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    start_time = time.time()
    
    # Get random indices
    total_samples = len(ds.time)
    random_indices = random.sample(range(total_samples), num_samples)

    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            sample_time = ds.time[idx]  # Rename the variable to `sample_time`
            date = pd.to_datetime(sample_time.values)
            data = np.stack([
                ds['u'].isel(time=idx).values,
                ds['v'].isel(time=idx).values,
                ds['r'].isel(time=idx).values,
                ds['vo'].isel(time=idx).values
            ])
            
            input_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)
            output = model(input_tensor)
            _, predicted = output.max(1)
            
            true_label = get_true_label(date, typhoon_data)
            
            all_preds.append(predicted.item())
            all_labels.append(true_label)
            
            total += 1
            correct += (predicted.item() == true_label)
            
            # Display progress and current accuracy every 10 samples
            if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                accuracy = 100 * correct / total
                elapsed_time = time.time() - start_time
                samples_per_second = (i + 1) / elapsed_time
                eta = (num_samples - (i + 1)) / samples_per_second

                print(f"Processed {i + 1}/{num_samples} samples "
                      f"({(i + 1)/num_samples*100:.2f}%) | "
                      f"Current accuracy: {accuracy:.2f}% | "
                      f"Samples/second: {samples_per_second:.2f} | "
                      f"ETA: {eta:.2f} seconds")
    
    print("Evaluation completed.")
    return all_preds, all_labels


def calculate_f1_scores(all_preds, all_labels):
    class_names = ['cyclogenesis', 'full typhoon', 'cyclolysis', 'no cyclone']
    f1_scores = {}
    
    for i, class_name in enumerate(class_names):
        true_positives = sum((np.array(all_labels) == i) & (np.array(all_preds) == i))
        false_positives = sum((np.array(all_labels) != i) & (np.array(all_preds) == i))
        false_negatives = sum((np.array(all_labels) == i) & (np.array(all_preds) != i))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[class_name] = f1
    
    # Calculate global F1-score
    f1_scores['global'] = np.mean(list(f1_scores.values()))
    
    return f1_scores

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix_random_sample.png')
    print("Confusion matrix saved as 'confusion_matrix_random_sample.png'")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading typhoon data...")
    typhoon_data = load_typhoon_data('typhoon_data_Cyclogenesis_Identification.csv')

    print("Loading NetCDF data...")
    ds = xr.open_dataset('/home/yazid/Documents/stage_cambridge/project_1/Pacific_Pressure_750.nc')

    print("Loading model...")
    model = StormCNET1(input_channels=4).to(device)
    model.load_state_dict(torch.load('storm_cnet1_model.pth', map_location=device))

    print("Evaluating model on 5000 random samples...")
    all_preds, all_labels = evaluate_model(model, ds, typhoon_data, device, num_samples=5000)

    print(f"\nEvaluation Results:")
    print(f"Total samples processed: {len(all_preds)}")
    final_accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_preds)
    print(f"Final Accuracy: {final_accuracy:.2f}%")

    f1_scores = calculate_f1_scores(all_preds, all_labels)
    print("\nF1-scores:")
    for class_name, f1 in f1_scores.items():
        print(f"  {class_name}: {f1:.4f}")

    class_names = ['cyclogenesis', 'full typhoon', 'cyclolysis', 'no cyclone']
    cm = np.zeros((4, 4), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1
    
    print("\nConfusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    main()