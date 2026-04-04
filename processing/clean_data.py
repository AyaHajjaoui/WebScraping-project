import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_raw_data():
    """Load all raw CSV files from data/raw directory"""
    raw_path = Path(__file__).parent.parent / "data" / "raw"
    
    files = {
        "openmeteo": raw_path / "openmeteo_raw.csv",
        "timeanddate": raw_path / "timeanddate_raw.csv",
        "wunderground": raw_path / "wunderground_raw.csv"
    }
    
    dataframes = {}
    for name, filepath in files.items():
        df = pd.read_csv(filepath)
        print(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
        dataframes[name] = df
    
    return dataframes

def standardize_datetime(dt_str):
    """Convert various datetime formats to standardized ISO format"""
    if pd.isna(dt_str) or dt_str == "":
        return None
    
    dt_str = str(dt_str).strip()
    
    try:
        # Handle ISO format with timezone (e.g., 2026-03-30T12:19:16.777393+00:00)
        if "+" in dt_str or dt_str.endswith("Z"):
            dt = pd.to_datetime(dt_str, utc=True)
        else:
            # Handle other formats
            dt = pd.to_datetime(dt_str)
        return dt.isoformat()
    except:
        return None

def clean_temperature(temp_value):
    """Clean temperature values - remove outliers that look like Fahrenheit"""
    if pd.isna(temp_value) or temp_value == "":
        return np.nan
    
    temp = float(temp_value)
    
    # If temperature > 40 and source is WeatherUnderground, likely Fahrenheit
    # Convert to Celsius
    if temp > 40:
        temp = (temp - 32) * 5/9
    
    # Range check: weather typically between -50 and 50 Celsius
    if temp < -50 or temp > 50:
        return np.nan
    
    return round(temp, 1)

def clean_numeric(value):
    """Clean numeric values"""
    if pd.isna(value) or value == "":
        return np.nan
    try:
        return float(value)
    except:
        return np.nan

def remove_high_missing_values(df, missing_threshold=4):
    """Remove rows with too many missing values"""
    # Count missing values per row
    missing_count = df.isna().sum(axis=1)
    # Keep rows with less than threshold missing columns
    df_filtered = df[missing_count < missing_threshold]
    rows_removed = len(df) - len(df_filtered)
    return df_filtered, rows_removed

def remove_key_duplicates(df):
    """Remove duplicates based on key columns"""
    key_columns = ['ScrapeDateTime', 'SourceWebsite', 'City', 'Country', 'Temperature_C']
    initial_rows = len(df)
    df_deduplicated = df.drop_duplicates(subset=key_columns, keep='first')
    rows_removed = initial_rows - len(df_deduplicated)
    return df_deduplicated, rows_removed

def fill_missing_numerical_values(df, source_name):
    """Fill missing numerical values with median for the source"""
    numerical_cols = ['FeelsLike_C', 'Humidity_%', 'WindSpeed_kmh']
    
    for col in numerical_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            # Calculate median for this source
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col].fillna(median_val, inplace=True)
                print(f"    Filled {missing_count} missing {col} values with median: {median_val:.2f}")
            else:
                # If all values are NaN, use mean
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    df[col].fillna(mean_val, inplace=True)
                    print(f"    Filled {missing_count} missing {col} values with mean: {mean_val:.2f}")
    
    return df

def clean_dataframes(dataframes):
    """Apply cleaning transformations to all dataframes"""
    cleaned = {}
    
    for name, df in dataframes.items():
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        print(f"\nCleaning {name}...")
        print(f"  Initial rows: {len(df_clean)}")
        
        # Standardize datetime
        df_clean['ScrapeDateTime'] = df_clean['ScrapeDateTime'].apply(standardize_datetime)
        
        # Clean numeric columns
        df_clean['Temperature_C'] = df_clean['Temperature_C'].apply(clean_temperature)
        df_clean['FeelsLike_C'] = df_clean['FeelsLike_C'].apply(clean_numeric)
        df_clean['Humidity_%'] = df_clean['Humidity_%'].apply(clean_numeric)
        df_clean['WindSpeed_kmh'] = df_clean['WindSpeed_kmh'].apply(clean_numeric)
        
        # Standardize text columns
        df_clean['Condition'] = df_clean['Condition'].fillna('').str.strip()
        df_clean['Condition'] = df_clean['Condition'].replace('', np.nan)
        
        # Remove rows with too many missing columns (4+)
        initial_rows = len(df_clean)
        df_clean, rows_removed = remove_high_missing_values(df_clean, missing_threshold=4)
        print(f"  After removing rows with 4+ missing columns: {len(df_clean)} rows (removed {rows_removed})")
        
        # Remove rows with null datetime (indicates bad record)
        initial_rows = len(df_clean)
        df_clean = df_clean[df_clean['ScrapeDateTime'].notna()]
        print(f"  After removing invalid datetimes: {len(df_clean)} rows (removed {initial_rows - len(df_clean)})")
        
        # Remove rows with no temperature data
        initial_rows = len(df_clean)
        df_clean = df_clean[df_clean['Temperature_C'].notna()]
        print(f"  After removing rows without temperature: {len(df_clean)} rows (removed {initial_rows - len(df_clean)})")
        
        # Remove duplicates based on key columns
        initial_rows = len(df_clean)
        df_clean, rows_removed = remove_key_duplicates(df_clean)
        print(f"  After removing key duplicates: {len(df_clean)} rows (removed {rows_removed})")
        
        # Fill missing numerical values with median/mean
        print(f"  Filling missing numerical values...")
        df_clean = fill_missing_numerical_values(df_clean, name)
        
        cleaned[name] = df_clean
    
    return cleaned

def combine_data(cleaned_dfs):
    """Combine all cleaned dataframes into one"""
    combined = pd.concat(list(cleaned_dfs.values()), ignore_index=True)
    print(f"\nCombined dataframe: {len(combined)} rows")
    
    # Remove exact duplicates
    initial_rows = len(combined)
    combined = combined.drop_duplicates()
    print(f"After removing exact duplicates: {len(combined)} rows (removed {initial_rows - len(combined)})")
    
    # Sort by datetime and city
    combined = combined.sort_values(['ScrapeDateTime', 'City']).reset_index(drop=True)
    
    # Reorder columns for better readability
    column_order = ['ScrapeDateTime', 'SourceWebsite', 'City', 'Country', 
                    'Temperature_C', 'FeelsLike_C', 'Humidity_%', 'WindSpeed_kmh', 
                    'Condition']
    combined = combined[column_order]
    
    return combined

def save_clean_data(df):
    """Save cleaned data to processed directory"""
    output_path = Path(__file__).parent.parent / "data" / "processed" / "weather_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Cleaned data saved to: {output_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Date range: {df['ScrapeDateTime'].min()} to {df['ScrapeDateTime'].max()}")
    print(f"\nData Summary:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())

def main():
    """Main cleaning pipeline"""
    print("=" * 60)
    print("CSV Data Cleaning Pipeline")
    print("=" * 60)
    
    # Load data
    print("\nLoading raw data...")
    dataframes = load_raw_data()
    
    # Clean data
    print("\nCleaning data...")
    cleaned_dfs = clean_dataframes(dataframes)
    
    # Combine data
    print("\nCombining data...")
    combined_df = combine_data(cleaned_dfs)
    
    # Save clean data
    print("\nSaving cleaned data...")
    save_clean_data(combined_df)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Data cleaning complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
