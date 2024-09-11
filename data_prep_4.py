import pandas as pd
import xarray as xr
import numpy as np
import os
import gc
import torch

def load_typhoon_data(file_path):
    print("Loading typhoon data...")
    typhoon_data = pd.read_csv(file_path)
    print("Typhoon data loaded.")
    return typhoon_data

def preprocess_typhoon_data(typhoon_data, start_date, end_date):
    typhoon_dict = {}
    for _, row in typhoon_data.iterrows():
        typhoon_name = row['Typhoon Name']
        cyclogenesis_start = pd.to_datetime(row['Cyclogenesis Start'])
        cyclogenesis_end = pd.to_datetime(row['Cyclogenesis End'])
        typhoon_start = pd.to_datetime(row['Typhoon Start'])
        typhoon_end = pd.to_datetime(row['Typhoon End'])
        cyclolysis_start = pd.to_datetime(row['Cyclolysis Start'])
        cyclolysis_end = pd.to_datetime(row['Cyclolysis End'])
        
        # Check if the typhoon is within the period
        if cyclogenesis_start > end_date or cyclolysis_end < start_date:
            continue
        
        for date in pd.date_range(max(cyclogenesis_start, start_date), min(cyclolysis_end, end_date), freq='6h'):
            if cyclogenesis_start <= date <= cyclogenesis_end:
                typhoon_dict[date] = (0, typhoon_name)  # cyclogenesis
            elif typhoon_start <= date <= typhoon_end:
                typhoon_dict[date] = (1, typhoon_name)  # full typhoon
            elif cyclolysis_start <= date <= cyclolysis_end:
                typhoon_dict[date] = (2, typhoon_name)  # cyclolysis
    return typhoon_dict

def process_and_save_chunk(chunk, typhoon_dict, start_index, total_samples, output_dir):
    X_chunk = []
    y_chunk = []
    for i, time in enumerate(chunk.time):
        date = pd.to_datetime(time.values)
        
        data = np.stack([
            chunk['u'].sel(time=time).values,
            chunk['v'].sel(time=time).values,
            chunk['r'].sel(time=time).values,
            chunk['vo'].sel(time=time).values
        ])
        
        label, typhoon_name = typhoon_dict.get(date, (3, None))  # 3 for no cyclone
        
        X_chunk.append(data)
        y_chunk.append(label)
        
        if (start_index + i + 1) % 100 == 0:
            progress = (start_index + i + 1) / total_samples * 100
            print(f"[{progress:.2f}%] Date: {date}, Class: {['cyclogenesis', 'full typhoon', 'cyclolysis', 'no cyclone'][label]}, Typhoon: {typhoon_name if typhoon_name else 'N/A'}")
    
    X_chunk = np.array(X_chunk)
    y_chunk = np.array(y_chunk)
    
    chunk_file = os.path.join(output_dir, f'chunk_{start_index}.npz')
    np.savez_compressed(chunk_file, X=X_chunk, y=y_chunk)
    
    print(f"Chunk saved. Chunk size: {X_chunk.shape[0]} samples")
    return X_chunk.shape[0], y_chunk

def main():
    print("Starting data preparation...")

    typhoon_data = load_typhoon_data('typhoon_data_Cyclogenesis_Identification.csv')

    ds = xr.open_dataset('/home/yazid/Documents/stage_cambridge/project_1/Pacific_Pressure_750.nc', chunks={'time': 800})

    start_date = pd.to_datetime(ds.time.values[0])
    end_date = start_date + pd.DateOffset(years=40)
    ds = ds.sel(time=slice(start_date, end_date))

    print(f"Period covered: from {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"Total number of samples in the dataset: {len(ds.time)}")
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Variables in the dataset: {list(ds.variables)}")

    typhoon_dict = preprocess_typhoon_data(typhoon_data, start_date, end_date)

    output_dir = 'processed_chunks_40_years'
    os.makedirs(output_dir, exist_ok=True)

    total_samples = len(ds.time)
    total_processed = 0
    all_y = []

    for i, chunk in enumerate(ds.chunks['time']):
        print(f"Processing chunk {i+1}...")
        chunk_data = ds.isel(time=slice(total_processed, total_processed + chunk))
        chunk_size, chunk_y = process_and_save_chunk(chunk_data, typhoon_dict, total_processed, total_samples, output_dir)
        
        all_y.extend(chunk_y)
        total_processed += chunk_size
        print(f"Progress: {total_processed}/{total_samples} samples processed ({total_processed/total_samples*100:.2f}%)")
        
        gc.collect()

    print(f"Total number of samples processed: {total_processed}")

    print("Class distribution:")
    unique, counts = np.unique(all_y, return_counts=True)
    for i, count in zip(unique, counts):
        class_name = ['cyclogenesis', 'full typhoon', 'cyclolysis', 'no cyclone'][i]
        print(f"Class {i} ({class_name}): {count} samples")

    chunk_files = sorted([f for f in os.listdir(output_dir) if f.startswith('chunk_') and f.endswith('.npz')])
    
    train_size = int(0.8 * total_processed)
    test_size = total_processed - train_size

    print("Saving information about prepared data...")
    torch.save({
        'train_size': train_size,
        'test_size': test_size,
        'chunk_files': chunk_files,
        'output_dir': output_dir
    }, 'cyclone_data_prepared_info.pt')

    print("Information about prepared data saved successfully.")
    print(f"Total number of samples: {total_processed}")
    print(f"Number of training samples: {train_size}")
    print(f"Number of test samples: {test_size}")

    print("Data preparation completed.")

if __name__ == "__main__":
    main()