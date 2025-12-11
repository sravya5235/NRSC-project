import numpy as np
import rasterio
from rasterio.plot import show
from skimage import filters, morphology
import cv2
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Polygon
import os
from tqdm import tqdm
from datetime import datetime
import math
import zipfile
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import psutil
from sklearn.model_selection import train_test_split
from cloud_shadow_detection import CloudShadowDetector, CloudShadowNet

class CloudShadowDataset(Dataset):
    """Dataset for cloud and shadow detection."""
    def __init__(self, features, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.transform:
            feature = self.transform(feature)
        
        if self.labels is not None:
            label = self.labels[idx]
            return torch.FloatTensor(feature), torch.LongTensor([label])
        else:
            return torch.FloatTensor(feature)

def prepare_training_data(detector, ground_truth_path=None):
    """Prepare training data from the image and optional ground truth."""
    print("[INFO] Preparing training data...")
    
    # Extract features from all tiles
    all_features = []
    all_labels = []
    
    height, width = detector.src.shape
    n_tiles_h = (height + detector.tile_size - 1) // detector.tile_size
    n_tiles_w = (width + detector.tile_size - 1) // detector.tile_size
    
    print(f"[INFO] Extracting features from {n_tiles_h * n_tiles_w} tiles...")
    
    for i in tqdm(range(n_tiles_h), desc="Processing rows"):
        for j in range(n_tiles_w):
            # Calculate tile bounds
            y_start = i * detector.tile_size
            y_end = min((i + 1) * detector.tile_size, height)
            x_start = j * detector.tile_size
            x_end = min((j + 1) * detector.tile_size, width)
            
            # Read tile
            tile_data = detector.src.read(window=((y_start, y_end), (x_start, x_end)))
            
            # Process tile
            tile_toa = detector.dn_to_toa_tile(tile_data)
            tile_features = detector.extract_features_tile(tile_toa)
            
            # Sample pixels for training (to avoid memory issues)
            sample_size = min(1000, tile_features.shape[0] * tile_features.shape[1])
            if sample_size > 0:
                # Randomly sample pixels
                indices = np.random.choice(
                    tile_features.shape[0] * tile_features.shape[1], 
                    sample_size, 
                    replace=False
                )
                
                for idx in indices:
                    row = idx // tile_features.shape[1]
                    col = idx % tile_features.shape[1]
                    feature = tile_features[row, col, :]
                    all_features.append(feature)
                    
                    # Generate synthetic labels for training (if no ground truth)
                    if ground_truth_path is None:
                        # Simple rule-based labeling for training
                        ndvi = feature[0]
                        ndbi = feature[1]
                        
                        if ndvi > 0.3 or ndbi > 0.1:
                            label = 1  # CLOUD
                        elif ndvi < -0.1 and ndbi < -0.1:
                            label = 2  # SHADOW
                        else:
                            label = 0  # NOCLOUD
                    else:
                        # Use actual ground truth
                        label = read_ground_truth_pixel(ground_truth_path, y_start + row, x_start + col)
                    
                    all_labels.append(label)
    
    print(f"[INFO] Prepared {len(all_features)} training samples")
    return np.array(all_features), np.array(all_labels)

def read_ground_truth_pixel(ground_truth_path, row, col):
    """Read a single pixel from ground truth image."""
    try:
        with rasterio.open(ground_truth_path) as src:
            return src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
    except:
        return 0  # Default to NOCLOUD if reading fails

def train_model(detector, features, labels, epochs=50, batch_size=32, validation_split=0.2):
    """Train the PyTorch model with the prepared data."""
    print("[INFO] Training model...")
    
    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=validation_split, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = CloudShadowDataset(X_train, y_train)
    val_dataset = CloudShadowDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    in_channels = features.shape[1]
    num_classes = 3
    detector.model = CloudShadowNet(in_channels, num_classes).to(detector.device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.model.parameters(), lr=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        detector.model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(detector.device)
            batch_labels = batch_labels.squeeze().to(detector.device)
            
            optimizer.zero_grad()
            outputs = detector.model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        detector.model.eval()
        val_loss = 0.0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(detector.device)
                batch_labels = batch_labels.squeeze().to(detector.device)
                
                outputs = detector.model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_precision = precision_score(val_true_labels, val_predictions, average='weighted')
        val_recall = recall_score(val_true_labels, val_predictions, average='weighted')
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(detector.model.state_dict(), 'best_model.pth')
            print("  [INFO] Best model saved!")
    
    # Load best model
    detector.model.load_state_dict(torch.load('best_model.pth'))
    print("[INFO] Training complete!")
    
    return train_losses, val_losses

def evaluate_model(detector, features, labels):
    """Evaluate the trained model and return comprehensive metrics."""
    print("[INFO] Evaluating model...")
    
    detector.model.eval()
    all_predictions = []
    all_labels = []
    
    # Create evaluation dataset
    eval_dataset = CloudShadowDataset(features, labels)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch_features, batch_labels in eval_loader:
            batch_features = batch_features.to(detector.device)
            batch_labels = batch_labels.squeeze().to(detector.device)
            
            outputs = detector.model(batch_features)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    
    # Overall metrics
    metrics['Overall_Accuracy'] = accuracy_score(all_labels, all_predictions)
    metrics['Overall_Precision'] = precision_score(all_labels, all_predictions, average='weighted')
    metrics['Overall_Recall'] = recall_score(all_labels, all_predictions, average='weighted')
    metrics['Overall_F1'] = f1_score(all_labels, all_predictions, average='weighted')
    
    # Per-class metrics
    class_names = ['NOCLOUD', 'CLOUD', 'SHADOW']
    for i, class_name in enumerate(class_names):
        y_true = (np.array(all_labels) == i)
        y_pred = (np.array(all_predictions) == i)
        
        metrics[f'{class_name}_Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{class_name}_Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{class_name}_F1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate IoU
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        metrics[f'{class_name}_IoU'] = intersection / union if union > 0 else 0
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['Overall_Accuracy']:.4f}")
    print(f"Overall F1-Score: {metrics['Overall_F1']:.4f}")
    print(f"Overall Precision: {metrics['Overall_Precision']:.4f}")
    print(f"Overall Recall: {metrics['Overall_Recall']:.4f}")
    print("\nPer-Class Metrics:")
    for class_name in class_names:
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics[f'{class_name}_Precision']:.4f}")
        print(f"  Recall: {metrics[f'{class_name}_Recall']:.4f}")
        print(f"  F1-Score: {metrics[f'{class_name}_F1']:.4f}")
        print(f"  IoU: {metrics[f'{class_name}_IoU']:.4f}")
    print("="*50)
    
    return metrics

def save_georeferenced_results(detector, output_dir):
    """Save the detection results as georeferenced GeoTIFF (8-bit) with proper class values."""
    print(f"[INFO] Saving georeferenced results to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined mask as georeferenced GeoTIFF (8-bit)
    mask_path = os.path.join(output_dir, 'cloud_shadow_mask.tiff')
    
    # Copy metadata from source image
    metadata = detector.metadata.copy()
    metadata.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': None
    })
    
    with rasterio.open(mask_path, 'w', **metadata) as dst:
        dst.write(detector.combined_mask.astype('uint8'), 1)
    
    print(f"[INFO] Georeferenced mask saved: {mask_path}")
    print("[INFO] Class values: NOCLOUD=0, CLOUD=1, SHADOW=2")
    
    # Save shapefiles
    def mask_to_shapefile(mask, output_path):
        contours, _ = cv2.findContours(
            mask.astype('uint8'),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            if len(contour) > 2:
                polygon = Polygon(contour.reshape(-1, 2))
                if polygon.area > 100:
                    polygons.append(polygon)
        
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=detector.metadata['crs'])
        gdf.to_file(output_path)
    
    # Save cloud and shadow shapefiles
    cloud_shp = os.path.join(output_dir, 'cloud.shp')
    shadow_shp = os.path.join(output_dir, 'shadow.shp')
    
    mask_to_shapefile(detector.cloud_mask, cloud_shp)
    mask_to_shapefile(detector.shadow_mask, shadow_shp)
    
    # Create zip files
    with zipfile.ZipFile(os.path.join(output_dir, 'cloudshapes.zip'), 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            zipf.write(cloud_shp.replace('.shp', ext), os.path.basename(cloud_shp.replace('.shp', ext)))
    
    with zipfile.ZipFile(os.path.join(output_dir, 'shadowshapes.zip'), 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            zipf.write(shadow_shp.replace('.shp', ext), os.path.basename(shadow_shp.replace('.shp', ext)))
    
    print("[INFO] Results saved.")
    return mask_path

def main():
    """Main function to train, evaluate, and process the image."""
    # Configuration
    input_path = "BAND4.tif"
    output_dir = "output"
    ground_truth_path = None  # Set to path if you have ground truth
    tile_size = 512
    epochs = 30  # Reduced for faster training
    
    # Load metadata (same as in run_detection.py)
    metadata = {
        'ProductID': 2552459121,
        'OTSProductID': 'RAF16MAY2025043798010000060SSANSTUC00GTDC',
        'SatID': 'IRS-R2A',
        'Sensor': 'L4FX',
        'SubScene': 'C',
        'GenAgency': 'NRSC',
        'Path': 100,
        'Row': 60,
        'SegmentNumber': None,
        'SessionNumber': 1,
        'StripNumber': None,
        'SceneNumber': None,
        'DateOfPass': '16-MAY-2025',
        'NoOfBands': 3,
        'BandNumbers': '234',
        'PassType': 'PLD',
        'DateOfDump': '16-MAY-2025',
        'DumpingOrbitNo': 43791,
        'ImagingOrbitNo': 43791,
        'BytesPerPixel': 2,
        'BitsPerPixel': 10,
        'GenerationDateTime': '18-MAY-2025 01:44:22',
        'ProdCode': 'STUC00GTD',
        'ProdType': 'GEOREF',
        'InputResolutionAlong': 5.80,
        'InputResolutionAcross': 5.80,
        'OutputResolutionAlong': 5.00,
        'OutputResolutionAcross': 5.00,
        'Season': 'MAY',
        'ImageFormat': 'GEOTIFF',
        'ProcessingLevel': 'STD',
        'ResampCode': 'CC',
        'NoScans': 16257,
        'NoPixels': 18067,
        'StartPixel': 0,
        'MapProjection': 'UTM',
        'Ellipsoid': 'WGS_84',
        'Datum': 'WGS84',
        'MapOriginLat': 0.0,
        'MapOriginLon': 81.0,
        'ProdULLat': 18.023387,
        'ProdULLon': 78.337935,
        'ProdURLat': 18.033245,
        'ProdURLon': 79.190687,
        'ProdLRLat': 17.298926,
        'ProdLRLon': 79.198016,
        'ProdLLLat': 17.289495,
        'ProdLLLon': 78.348709,
        'ImageULLat': 18.025828,
        'ImageULLon': 78.521608,
        'ImageURLat': 18.033252,
        'ImageURLon': 79.190678,
        'ImageLRLat': 17.297198,
        'ImageLRLon': 79.015029,
        'ImageLLLat': 17.289504,
        'ImageLLLon': 78.348701,
        'ProdULMapX': 218141.443084,
        'ProdULMapY': 1994800.0,
        'ProdURMapX': 308471.443084,
        'ProdURMapY': 1994800.0,
        'ProdLRMapX': 308471.443084,
        'ProdLRMapY': 1913520.0,
        'ProdLLMapX': 218141.443084,
        'ProdLLMapY': 1913520.0,
        'SceneCenterLat': 17.66173,
        'SceneCenterLon': 78.768824,
        'StandardParallel1': None,
        'StandardParallel2': None,
        'FalseEasting': 500000.0,
        'FalseNorthing': 0.0,
        'ZoneNo': 44,
        'SceneCenterTime': '16-MAY-2025 05:12:05.800191',
        'SceneCenterRoll': -0.023069,
        'SceneCenterPitch': -0.030934,
        'SceneCenterYaw': 3.640214,
        'SunAzimuthAtCenter': 82.426782,
        'SunElevationAtCenter': 68.777133,
        'ImageHeadingAngle': 193.30371,
        'IncidenceAngle': 0.765154,
        'SatelliteAltitude': 821.387047,
        'Tiltangle': -0.72966,
        'DEMCorrection': 'YES',
        'SourceOfOrbit': 2,
        'SourceOfAttitude': 1,
        'ImagingDirection': 'D',
        'B2Temp': 17.93,
        'B3Temp': 18.37,
        'B4Temp': 17.04,
        'B2_Lmin': 0.0,
        'B3_Lmin': 0.0,
        'B4_Lmin': 0.0,
        'B2_Lmax': 52.0,
        'B3_Lmax': 47.0,
        'B4_Lmax': 31.5,
        'Quality': 'Q',
        'CloudPercent': None,
        'AcrossTrackAccuracy': 0.0,
        'AlongTrackAccuracy': 0.0,
        'Shift%': 0,
        'SatelliteHeadingAngle': 193.30371,
        'SceneStartTime': '16-MAY-2025 05:11:59.500388084',
        'SceneEndTime': '16-MAY-2025 05:12:10.039701254',
        'SenAzimuthAtCenter': 89.068787,
        'SenElevationAtCenter': 89.068787,
        'IncidenceAngleAlongTrack': 0.896035,
        'IncidenceAngleAcrossTrack': -0.324528,
        'ViewAngle': 0.677829,
        'ViewAngleAlongTrack': -0.631071,
        'ViewAngleAcrossTrack': -0.24741,
        'ProductSceneStartTime': '16-MAY-2025 05:11:58.661205',
        'ProductSceneEndTime': '16-MAY-2025 05:12:12.939177',
        'JpegNoColumns': 512,
        'JpegNoRows': 569
    }
    
    print("="*60)
    print("CLOUD AND SHADOW DETECTION - TRAINING AND EVALUATION")
    print("="*60)
    
    # Initialize detector
    detector = CloudShadowDetector(input_path, metadata, tile_size)
    detector.load_image()
    
    # Prepare training data
    features, labels = prepare_training_data(detector, ground_truth_path)
    
    # Train model
    train_losses, val_losses = train_model(detector, features, labels, epochs=epochs)
    
    # Evaluate model
    metrics = evaluate_model(detector, features, labels)
    
    # Save evaluation results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
    print(f"[INFO] Evaluation metrics saved to {output_dir}/evaluation_metrics.csv")
    
    # Process the full image with trained model
    print("\n[INFO] Processing full image with trained model...")
    detector.process_image_tiled()
    
    # Save georeferenced results
    mask_path = save_georeferenced_results(detector, output_dir)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print(f"Output files saved to: {output_dir}")
    print(f"Georeferenced mask: {mask_path}")
    print("Class values: NOCLOUD=0, CLOUD=1, SHADOW=2")
    print("="*60)

if __name__ == "__main__":
    main() 