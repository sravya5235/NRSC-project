from cloud_shadow_detection import process_image
import os

# Set your input and output paths
input_path = "BAND4.tif"  # Updated to use BAND4.tif as input
output_dir = "output"

# Option to process only a subset for testing (set to True for faster testing)
TEST_MODE = False  # Set to False for full image processing
TEST_SIZE = 1000  # Size of test subset (e.g., 1000x1000 pixels)

# Tile size for processing large images (smaller = less memory, slower)
TILE_SIZE = 512  # Adjust based on your system's memory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Example metadata (replace with your actual values)
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
print("CLOUD AND SHADOW DETECTION PIPELINE")
print("="*60)
print(f"Input: {input_path}")
print(f"Output: {output_dir}")
print(f"Tile Size: {TILE_SIZE}")
print("="*60)

# Process the image
detector = process_image(
    input_path=input_path,
    output_dir=output_dir,
    metadata=metadata,
    tile_size=TILE_SIZE,
    train_model=False  # Disable training to avoid memory issues
)

print(f"\nProcessing complete. Results saved to {output_dir}")
print("Generated files:")
print("- mask.tiff: Combined cloud and shadow mask (8-bit)")
print("- cloudshapes.zip: Cloud polygons")
print("- shadowshapes.zip: Shadow polygons")
print("\nClass values in mask.tiff:")
print("- 0: No Cloud")
print("- 1: Cloud")
print("- 2: Shadow")
print("="*60) 