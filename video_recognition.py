import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight

# --- 1. DATA LOADER (Unchanged) ---
class VideoLoader:
    def __init__(self, directory, frame_size, frame_rate_scaler, classes_to_use=None):
        self.directory = directory
        self.db = []
        self.classes = [] 
        self.class_to_idx = {} 
        self.frame_size = frame_size
        self.frame_rate_scaler = frame_rate_scaler
        self.classes_to_use = classes_to_use 
        self.load_dataset()
        
    def load_dataset(self):
        if not os.path.exists(self.directory):
            print(f"Error: Directory '{self.directory}' not found.")
            return

        all_classes = sorted([d for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, d))])
        
        if self.classes_to_use is not None:
            self.classes = sorted([c for c in all_classes if c in self.classes_to_use])
        else:
            self.classes = all_classes

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Classes loaded: {self.classes}")

        for label in self.classes:
            folder_path = os.path.join(self.directory, label)
            video_files = sorted(glob.glob(os.path.join(folder_path, "*.avi")))
            for video_file in video_files:
                self.db.append((video_file, label))
                
        print(f"Database size: {len(self.db)}")

    def load_video(self, video_path, resize, n):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame_count % n == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if resize:
                        frame = cv2.resize(frame, resize)
                    frames.append(frame)
                frame_count += 1
        finally:
            cap.release()
        return np.array(frames)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        video_path, label_str = self.db[idx]
        frames = self.load_video(video_path, None, self.frame_rate_scaler) # video doesn't need to be resized anymore
        if len(frames) == 0:
            frames = torch.zeros((16, 3, self.frame_size, self.frame_size), dtype=torch.float32)
        else:
            frames = torch.tensor(frames, dtype=torch.float32)
            frames = frames.permute(0, 3, 1, 2)
            frames = frames / 255.0
        label_idx = self.class_to_idx[label_str]
        return frames, label_idx

# --- 2. PERSISTENCE HELPER (Unchanged) ---
def get_persistent_splits(dataset, ratio, save_prefix):
    train_path = f"{save_prefix}_train.pkl"
    test_path = f"{save_prefix}_test.pkl"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Loading existing split from {save_prefix}...")
        with open(train_path, 'rb') as f:
            train_set = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_set = pickle.load(f)
    else:
        print(f"Creating NEW split for {save_prefix}...")
        train_len = int(ratio * len(dataset))
        test_len = len(dataset) - train_len
        train_set, test_set = random_split(dataset, [train_len, test_len])
        
        with open(train_path, 'wb') as f:
            pickle.dump(train_set, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_set, f)
        print(f"Split saved to {train_path} and {test_path}")
        
    return train_set, test_set

# --- 3. MODEL ARCHITECTURE (Unchanged) ---
MAX_POOL = 1
AVG_POOL = 0

class CNN(nn.Module):
    def __init__(self, layer_config, poolType, input_dims, embedding_dim):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = input_dims[0]
        current_h, current_w = input_dims[1], input_dims[2]

        for i, config in enumerate(layer_config):
            out_ch = config['out_channels']
            k = config['kernel_size']
            s = config['stride']
            p = config['padding']
            
            if(poolType == MAX_POOL):
                layer = nn.Sequential(
                    nn.Conv2d(current_channels, out_ch, k, s, p),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(current_channels, out_ch, k, s, p),
                    nn.ReLU(),
                    nn.AvgPool2d(2, 2)
                )
            self.layers.append(layer)
            current_h = int((current_h + 2*p - k) / s) + 1
            current_w = int((current_w + 2*p - k) / s) + 1     
            current_h = int((current_h - 2) / 2) + 1
            current_w = int((current_w - 2) / 2) + 1
            current_channels = out_ch

        self.flatten_size = current_channels * current_h * current_w
        self.fc = nn.Linear(self.flatten_size, embedding_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, cnn_model, num_classes, lstm_hidden_size, lstm_layers):
        super(CNNLSTM, self).__init__()
        self.cnn = cnn_model
        self.cnn_output_size = cnn_model.fc.out_features 
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size, 
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        cnn_features = []
        for t in range(time_steps):
            frame = x[:, t, :, :, :] 
            frame_feature = self.cnn(frame)
            cnn_features.append(frame_feature)
            
        lstm_input = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(lstm_input)
        last_output = lstm_out[:, -1, :] 
        last_output = self.dropout(last_output)
        prediction = self.fc(last_output)
        return prediction

# --- 4. TRAIN/SAVE UTILS (UPDATED) ---
def save_model(model, model_file):  
    with open(model_file, 'wb') as f:
        pickle.dump(model, f) 
        print(f"Model saved as {model_file}")

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def train(model, epochs, accumulation_steps, learning_rate, train_loader, device, use_weighted_loss=False):
    # --- LOSS CONFIGURATION ---
    if use_weighted_loss:
        print("Calculating class weights for Weighted Loss...")
        
        # 1. Get labels from dataset efficiently
        dataset = train_loader.dataset
        if hasattr(dataset, 'indices'): # It's a Subset
            indices = dataset.indices
            source_dataset = dataset.dataset
        else: # It's the full VideoLoader
            indices = range(len(dataset))
            source_dataset = dataset
            
        all_labels = []
        for i in indices:
            _, label_str = source_dataset.db[i]
            label_idx = source_dataset.class_to_idx[label_str]
            all_labels.append(label_idx)
            
        # 2. Compute weights for PRESENT classes
        present_classes = np.unique(all_labels)
        computed_weights = compute_class_weight(
            class_weight='balanced', 
            classes=present_classes, 
            y=all_labels
        )
        
        # 3. Create Full Weight Tensor (handling missing classes)
        # Get total number of classes from the model's output layer
        total_num_classes = model.fc.out_features
        full_weights = torch.ones(total_num_classes, dtype=torch.float) # Default weight 1.0
        
        # Fill in the computed weights for the classes we actually have
        for cls_idx, weight in zip(present_classes, computed_weights):
            full_weights[cls_idx] = weight
            
        full_weights = full_weights.to(device)
        print(f"Class Weights (Shape {full_weights.shape}): {full_weights}")
        
        criterion = nn.CrossEntropyLoss(weight=full_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        interval_loss = 0.0
        
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            loss_val = loss.item() * accumulation_steps
            running_loss += loss_val
            interval_loss += loss_val
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 20 == 0:
                print(f"Epoch {epoch+1} Step [{i+1}/{len(train_loader)}] Loss: {interval_loss/20:.4f}")
                interval_loss = 0.0
        
        print(f"Epoch {epoch+1} Finished | Acc: {100 * correct / total:.2f}% | Loss: {running_loss/len(train_loader):.4f}")

def replace_head_for_finetuning(model, new_num_classes):
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_size = model.lstm.hidden_size
    model.fc = nn.Linear(hidden_size, new_num_classes)
    
    print(f"Model modified. CNN+LSTM frozen. New head created for {new_num_classes} classes.")
    return model