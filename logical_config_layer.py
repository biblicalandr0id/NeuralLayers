import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

class LogicalConfigurationLayer(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=24, output_dim=8):
        super(LogicalConfigurationLayer, self).__init__()
        
        # Layer for Configuration State (Cₖ)
        self.config_layer = nn.Linear(input_dim, hidden_dim)
        
        # Layer for Logical Reasoning (Lₖ)
        self.logic_layer = nn.Linear(input_dim, hidden_dim)
        
        # Combined processing layer (LC)
        self.combined_layer = nn.Linear(hidden_dim * 2, output_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
    def encode_timestamp(self, timestamp_str):
        """Encode timestamp into normalized features"""
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return np.array([
            dt.year / 2100,  # Normalize year
            dt.month / 12,   # Normalize month
            dt.day / 31,     # Normalize day
            dt.hour / 24,    # Normalize hour
            dt.minute / 60,  # Normalize minute
            dt.second / 60   # Normalize second
        ])
    
    def encode_username(self, username):
        """Encode username into numerical features"""
        # Convert username to numerical representation
        ascii_vals = [ord(c) / 255 for c in username[:6]]  # First 6 chars
        while len(ascii_vals) < 6:  # Pad if needed
            ascii_vals.append(0)
        return np.array(ascii_vals)
    
    def forward(self, x):
        """
        Forward pass implementing LC = (Cₖ ∧ Lₖ) → Cₖ₊₁
        x should contain both timestamp and username features
        """
        # Configuration State Processing (Cₖ)
        config_state = self.relu(self.config_layer(x))
        
        # Logical Reasoning Processing (Lₖ)
        logic_state = self.relu(self.logic_layer(x))
        
        # Combine states (Cₖ ∧ Lₖ)
        combined = torch.cat((config_state, logic_state), dim=1)
        combined = self.batch_norm(combined)
        
        # Produce next configuration state (Cₖ₊₁)
        output = self.sigmoid(self.combined_layer(combined))
        
        return output

    def process_input(self, timestamp_str, username):
        """Process raw timestamp and username into network input"""
        # Encode inputs
        time_features = self.encode_timestamp(timestamp_str)
        user_features = self.encode_username(username)
        
        # Combine features
        combined_features = np.concatenate([time_features, user_features])
        return torch.FloatTensor(combined_features)
