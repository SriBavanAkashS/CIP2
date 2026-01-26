import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==================================================
# MODULE 3: SPATIAL FEATURE LEARNING (CNN)
# ==================================================

OUTPUT_DIR = "../outputs/module3_spatial"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# SPATIAL CNN MODEL
# --------------------------------------------------
class SpatialEEGCNN(nn.Module):
    def __init__(self, num_channels=32, window_size=256, output_features=128):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 5),
            padding=(1, 2)
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 5),
            padding=(1, 2)
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        
        # Calculate flattened size after conv layers
        # Input: (batch, 1, num_channels, window_size)
        # After conv1 + pool1: (batch, 16, num_channels//2, window_size//2)
        # After conv2 + pool2: (batch, 32, num_channels//4, window_size//4)
        # Flattened: 32 * (num_channels//4) * (window_size//4)
        # Use integer division to handle any size
        conv_output_size = 32 * (num_channels // 4) * (window_size // 4)
        
        # Ensure minimum size (handle edge cases)
        if conv_output_size <= 0:
            raise ValueError(f"Invalid conv_output_size: {conv_output_size}. "
                           f"num_channels={num_channels}, window_size={window_size}. "
                           f"Both must be >= 4.")
        
        # Projection layer to get desired output size
        self.fc = nn.Linear(conv_output_size, output_features)
        self.relu3 = nn.ReLU()
        
        # Store for reference
        self.conv_output_size = conv_output_size

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc(x))
        return x


# --------------------------------------------------
# LOAD MODULE 2 OUTPUT
# --------------------------------------------------
def load_module2_output(path):
    """
    Load preprocessed data from Module 2
    
    Args:
        path: Path to the .npy file containing preprocessed data
    
    Returns:
        numpy array of shape (subjects, trials, segments, channels, window_size)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module 2 output not found at: {path}")
    
    data = np.load(path)
    print(f"Loaded Module 2 data shape: {data.shape}")
    
    if len(data.shape) != 5:
        raise ValueError(f"Expected 5D array (subjects, trials, segments, channels, window), "
                         f"got shape: {data.shape}")
    
    return data


# --------------------------------------------------
# SAVE/LOAD MODEL
# --------------------------------------------------
def save_model(model, path=None):
    """Save the CNN model to disk"""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "spatial_cnn_model.pth")
    
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")
    return path


def load_model(num_channels=32, window_size=256, output_features=128, path=None):
    """Load a saved CNN model from disk"""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "spatial_cnn_model.pth")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    
    model = SpatialEEGCNN(num_channels=num_channels, window_size=window_size, 
                          output_features=output_features)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from: {path}")
    return model


# --------------------------------------------------
# EXTRACT SPATIAL FEATURES (for use by other modules)
# --------------------------------------------------
def extract_spatial_features(eeg_segments, return_model=False, model=None):
    """
    Extract spatial features from preprocessed EEG segments
    
    Args:
        eeg_segments: numpy array of shape (segments, channels, window_size)
                     or (subjects, trials, segments, channels, window_size)
        return_model: if True, return the model along with features
        model: Optional pre-initialized model. If None, creates a new one.
    
    Returns:
        spatial_features: numpy array of shape (segments, 128)
        model (optional): the CNN model
    """
    # Handle different input shapes
    original_shape = eeg_segments.shape
    
    if len(original_shape) == 5:
        # (subjects, trials, segments, channels, window)
        subjects, trials, segments, channels, window = original_shape
        X = eeg_segments.reshape(-1, channels, window)
    elif len(original_shape) == 3:
        # (segments, channels, window)
        X = eeg_segments
        channels, window = original_shape[1], original_shape[2]
    else:
        raise ValueError(f"Unexpected input shape: {original_shape}. "
                         f"Expected 3D (segments, channels, window) or "
                         f"5D (subjects, trials, segments, channels, window)")
    
    # Validate dimensions
    if channels < 4 or window < 4:
        raise ValueError(f"Invalid dimensions: channels={channels}, window={window}. "
                         f"Both must be >= 4 for the CNN architecture.")
    
    X = torch.tensor(X, dtype=torch.float32)
    X = X.unsqueeze(1)  # (batch, 1, channels, time)
    
    # Use provided model or create new one
    if model is None:
        model = SpatialEEGCNN(num_channels=channels, window_size=window, output_features=128)
    
    model.eval()
    
    with torch.no_grad():
        spatial_features = model(X)
    
    spatial_features = spatial_features.numpy()
    
    if return_model:
        return spatial_features, model
    return spatial_features


# --------------------------------------------------
# RUN MODULE 3 (for standalone execution)
# --------------------------------------------------
def run_module3(D_prep_all, batch_size=256):
    """
    Run Module 3: Extract spatial features using CNN
    
    Args:
        D_prep_all: numpy array of shape (subjects, trials, segments, channels, window_size)
        batch_size: Number of segments to process at once (default: 256)
                    Reduce if you run out of memory
    
    Returns:
        model: The trained CNN model
        X_sample: Sample input tensor (for visualization)
        spatial_features: numpy array of extracted features (batch, 128)
    """
    if len(D_prep_all.shape) != 5:
        raise ValueError(f"Expected 5D input (subjects, trials, segments, channels, window), "
                         f"got shape: {D_prep_all.shape}")
    
    subjects, trials, segments, channels, window = D_prep_all.shape
    
    # Validate dimensions
    if channels < 4 or window < 4:
        raise ValueError(f"Invalid dimensions: channels={channels}, window={window}. "
                         f"Both must be >= 4 for the CNN architecture.")

    # Reshape → each segment is one CNN sample
    X_all = D_prep_all.reshape(-1, channels, window)
    total_segments = X_all.shape[0]
    
    # Keep a sample for visualization (first segment)
    X_sample = torch.tensor(X_all[0:1], dtype=torch.float32).unsqueeze(1)

    # Initialize model
    model = SpatialEEGCNN(num_channels=channels, window_size=window, output_features=128)
    model.eval()

    print(f"Processing {total_segments} segments in batches of {batch_size}...")
    
    # Process in batches to avoid memory issues
    all_features = []
    
    with torch.no_grad():
        for i in range(0, total_segments, batch_size):
            batch_end = min(i + batch_size, total_segments)
            batch_X = X_all[i:batch_end]
            
            # Convert to tensor and add channel dimension
            batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32)
            batch_X_tensor = batch_X_tensor.unsqueeze(1)  # (batch, 1, channels, time)
            
            # Process batch
            batch_features = model(batch_X_tensor)
            all_features.append(batch_features.numpy())
            
            # Progress update
            if (i // batch_size + 1) % 10 == 0 or batch_end == total_segments:
                print(f"  Processed {batch_end}/{total_segments} segments "
                      f"({100*batch_end/total_segments:.1f}%)")

    # Concatenate all batches
    spatial_features = np.concatenate(all_features, axis=0)
    
    print(f"Module 3 Output Shape: {spatial_features.shape}")

    # Save features
    output_path = os.path.join(OUTPUT_DIR, "spatial_features.npy")
    np.save(output_path, spatial_features)
    print(f"Saved spatial features to: {output_path}")

    return model, X_sample, spatial_features


# --------------------------------------------------
# VISUALIZATION 1: CONV1 FEATURE MAP
# --------------------------------------------------
def visualize_conv1_feature_map(model, sample):
    """
    Visualize the first convolutional layer's feature map
    
    Args:
        model: The CNN model
        sample: Input sample tensor of shape (channels, time) or (1, channels, time) or (1, 1, channels, time)
    """
    # Ensure sample has correct shape
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32)
    
    # Normalize to (1, 1, channels, time) format
    if sample.dim() == 2:
        # (channels, time) -> (1, 1, channels, time)
        sample = sample.unsqueeze(0).unsqueeze(0)
    elif sample.dim() == 3:
        # (1, channels, time) -> (1, 1, channels, time)
        if sample.shape[0] == 1:
            sample = sample.unsqueeze(0)
        else:
            sample = sample.unsqueeze(0).unsqueeze(0)
    elif sample.dim() == 4:
        # (batch, 1, channels, time) -> take first sample
        sample = sample[0:1]
    else:
        raise ValueError(f"Unexpected sample shape: {sample.shape}")
    
    # Ensure it's (1, 1, channels, time)
    if sample.dim() != 4 or sample.shape[0] != 1 or sample.shape[1] != 1:
        raise ValueError(f"Sample must be (1, 1, channels, time), got {sample.shape}")
    
    with torch.no_grad():
        # Get conv1 output: (1, 16, channels, time)
        fmap = model.conv1(sample)
    
    # fmap shape should be (1, 16, channels, time)
    # Take first filter's output: fmap[0, 0] should give (channels, time)
    # But if we get 1D, try different indexing
    
    if fmap.dim() == 4:
        # Standard case: (batch, filters, channels, time)
        fmap_np = fmap[0, 0].cpu().numpy()  # Take first batch, first filter
    elif fmap.dim() == 3:
        # If batch dimension was squeezed: (filters, channels, time)
        fmap_np = fmap[0].cpu().numpy()  # Take first filter
    else:
        raise ValueError(f"Unexpected conv1 output dimensions: {fmap.dim()}, shape: {fmap.shape}")
    
    # Ensure it's 2D
    if fmap_np.ndim == 1:
        # If somehow 1D, reshape based on expected dimensions
        expected_channels, expected_time = sample.shape[2], sample.shape[3]
        if fmap_np.shape[0] == expected_channels * expected_time:
            fmap_np = fmap_np.reshape(expected_channels, expected_time)
        elif fmap_np.shape[0] == expected_time:
            # Might be just time dimension, need to add channel dimension
            fmap_np = fmap_np.reshape(1, expected_time)
        else:
            raise ValueError(f"Cannot reshape 1D feature map of shape {fmap_np.shape}. "
                           f"Expected channels={expected_channels}, time={expected_time}")
    elif fmap_np.ndim == 2:
        # Check if dimensions are correct (channels, time)
        expected_channels, expected_time = sample.shape[2], sample.shape[3]
        if fmap_np.shape[0] != expected_channels or fmap_np.shape[1] != expected_time:
            # Might be transposed or different order
            if fmap_np.shape[1] == expected_channels and fmap_np.shape[0] == expected_time:
                fmap_np = fmap_np.T
            # If still doesn't match, use as-is (might be due to padding/conv effects)
    else:
        raise ValueError(f"Feature map should be 1D or 2D after indexing, got shape {fmap_np.shape}")

    plt.figure(figsize=(12, 5))
    # Use RdBu (Red-Blue diverging) colormap to better show positive/negative activations
    # Red = positive activation, Blue = negative activation, White = zero
    plt.imshow(fmap_np, aspect="auto", cmap="RdBu_r", interpolation='nearest')
    plt.colorbar(label="Activation Strength")
    plt.title("Conv1 Spatial Feature Map (Channel × Time)", fontsize=14, fontweight='bold')
    plt.xlabel("Time Samples", fontsize=12)
    plt.ylabel("EEG Channels", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "conv1_feature_map.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ conv1_feature_map.png saved (shape: {fmap_np.shape})")


# --------------------------------------------------
# VISUALIZATION 2: FEATURE VECTOR DISTRIBUTION
# --------------------------------------------------
def visualize_feature_distribution(features):
    # Handle both torch tensors and numpy arrays
    if isinstance(features, torch.Tensor):
        vec = features[0].cpu().numpy()
    else:
        vec = features[0]

    plt.figure(figsize=(8, 4))
    plt.hist(vec, bins=50)
    plt.title("Spatial Feature Vector Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_distribution.png"), dpi=300)
    plt.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    print("\n" + "="*60)
    print("MODULE 3: SPATIAL FEATURE LEARNING (CNN)")
    print("="*60)

    # Load Module 2 output
    module2_output_path = "../outputs/module2_preprocessing/preprocessed_all_subjects.npy"
    D_prep_all = load_module2_output(module2_output_path)

    # Extract dimensions
    subjects, trials, segments, channels, window = D_prep_all.shape
    total_segments = subjects * trials * segments
    
    print(f"\nInput Data Summary:")
    print(f"  - Subjects: {subjects}")
    print(f"  - Trials per subject: {trials}")
    print(f"  - Segments per trial: {segments}")
    print(f"  - EEG Channels: {channels}")
    print(f"  - Window Size: {window} samples")
    print(f"  - Total segments to process: {total_segments}")

    # Run Spatial CNN
    print(f"\n{'='*60}")
    print("Extracting Spatial Features with CNN...")
    print(f"{'='*60}")
    
    # Process in batches to avoid memory issues
    # Adjust batch_size if you encounter memory errors (try 128, 64, or 32)
    model, X_sample, spatial_features = run_module3(D_prep_all, batch_size=256)

    # Save model
    save_model(model)

    # Visualizations
    print(f"\n{'='*60}")
    print("Generating Visualizations...")
    print(f"{'='*60}")
    
    visualize_conv1_feature_map(model, X_sample[0])
    visualize_feature_distribution(spatial_features)

    # Summary
    print(f"\n{'='*60}")
    print("✓ MODULE 3 COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nOutput Summary:")
    print(f"  - Input shape: {D_prep_all.shape}")
    print(f"  - Output shape: {spatial_features.shape}")
    print(f"  - Feature dimension: {spatial_features.shape[1]}")
    print(f"  - Total feature vectors: {spatial_features.shape[0]}")
    
    print(f"\nSaved Files:")
    print(f"  ✓ spatial_features.npy")
    print(f"  ✓ spatial_cnn_model.pth")
    print(f"  ✓ conv1_feature_map.png")
    print(f"  ✓ feature_distribution.png")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
