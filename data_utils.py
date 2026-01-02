import pandas as pd
import numpy as np
import re
import os
import streamlit as st

# Define hardcoded absolute paths for your specific environment
ABS_BASE_PATH = r"C:\Users\AkshayKarthickMS\Desktop\Phase-1-main\data"

def find_data_file(filename):
    """
    Robustly finds a data file by checking:
    1. Exact path provided.
    2. Common variations (singular/plural, csv/xlsx).
    3. The hardcoded absolute folder.
    """
    if not filename: return None
    
    # Clean filename and get base/ext
    clean_name = os.path.basename(filename)
    name_no_ext, ext = os.path.splitext(clean_name)
    
    # Variations to try (e.g., facility_visits -> facility_visit)
    name_variations = [
        name_no_ext,
        name_no_ext.rstrip('s'), # try singular
        name_no_ext + 's'        # try plural
    ]
    
    # Extensions to try
    extensions = [ext, '.csv', '.xlsx']
    
    # Search locations
    search_dirs = [
        os.path.dirname(filename),
        ABS_BASE_PATH,
        "data",
        ".",
        ""
    ]
    
    for folder in search_dirs:
        if folder is None: continue
        for name in name_variations:
            for try_ext in extensions:
                candidate = os.path.join(folder, name + try_ext)
                if os.path.exists(candidate):
                    return candidate
    return None

def parse_age_string(age_str):
    """Parses strings like '1year, 3months' into total months."""
    if not isinstance(age_str, str) or not age_str.strip():
        return 0.0
    s = age_str.lower().replace(',', '')
    years = float(re.search(r'(\d+)\s*y', s).group(1)) if re.search(r'(\d+)\s*y', s) else 0
    months = float(re.search(r'(\d+)\s*m', s).group(1)) if re.search(r'(\d+)\s*m', s) else 0
    weeks = float(re.search(r'(\d+)\s*w', s).group(1)) if re.search(r'(\d+)\s*w', s) else 0
    return (years * 12) + months + (weeks / 4.345)

AGE_CATEGORIES = ["Newborn (<6w)", "Due for Penta", "Active ZD", "Overaged ZD", "Older Child (>5y)"]

def classify_age_months(m):
    """
    Standard Age Classification Logic:
    - Newborn: < 1.5 months (< 6 weeks)
    - Due for Penta: 1.5 months - 11 months
    - Active ZD: 12 months - 23 months
    - Overaged ZD: 24 months - 5 years (60 months)
    - Older Child: > 5 years
    """
    if str(m) == 'nan': return "Unknown"
    if m < 1.5: return "Newborn (<6w)"
    if 1.5 <= m < 12: return "Due for Penta"
    if 12 <= m < 24: return "Active ZD"
    if 24 <= m < 60: return "Overaged ZD"
    return "Older Child (>5y)"

def apply_client_age_categories(df, source_type='facility'):
    """
    Applies Client Logic: >6w-11m (Penta), 12-23m (Active), 24m-5y (Overaged).
    Handles the specific columns seen in your screenshots (age_years, age_month, etc).
    """
    # 1. Normalize cols for detection
    df_cols = {c.lower(): c for c in df.columns}
    
    # 2. Detect Age Columns
    y_col = df_cols.get('age_years') or df_cols.get('current_age_years')
    m_col = df_cols.get('age_month') or df_cols.get('age_months') or df_cols.get('current_age_month')
    w_col = df_cols.get('age_weeks') or df_cols.get('current_age_weeks')

    # 3. Calculate Months
    # If explicit columns exist and are numeric-like
    if y_col and m_col:
        df['age_total_months'] = (
            pd.to_numeric(df[y_col], errors='coerce').fillna(0) * 12 +
            pd.to_numeric(df[m_col], errors='coerce').fillna(0) +
            (pd.to_numeric(df[w_col], errors='coerce').fillna(0) / 4.345 if w_col else 0)
        )
    else:
        # Fallback to string parsing if numeric cols missing
        col = next((c for c in df.columns if 'estimated' in c.lower() or 'enrollment' in c.lower()), None)
        if col:
            df['age_total_months'] = df[col].astype(str).apply(parse_age_string)
        else:
            df['age_total_months'] = 0.0

    # 4. Classify using shared function
    df['Client_Age_Group'] = df['age_total_months'].apply(classify_age_months)
    df['Client_Age_Group'] = pd.Categorical(df['Client_Age_Group'], categories=AGE_CATEGORIES, ordered=True)
    return df

def get_unique_lgas(filename):
    """
    Reads the file to get actual LGAs present.
    """
    path = find_data_file(filename)
    if not path:
        return ["Gabasawa LGA", "Ungogo LGA", "Kiru LGA"] # Fallback
    
    try:
        if path.endswith('.xlsx'): df = pd.read_excel(path)
        else: df = pd.read_csv(path)
        
        # Find LGA column
        lga_col = next((c for c in df.columns if 'lga' in c.lower() and 'id' not in c.lower()), None)
        if lga_col:
            return sorted(df[lga_col].dropna().unique().tolist())
    except:
        pass
    return ["Gabasawa LGA", "Ungogo LGA", "Kiru LGA"]

def filter_data(df, selected_lgas=None, start_date=None, end_date=None, date_col=None, selected_genders=None, include_na_dates=False):
    """Applies Filters Robustly with Smart Defaults for Future Data"""
    if df.empty: return df

    # 1. LGA Filter (Flexible)
    if selected_lgas:
        lga_col = next((c for c in df.columns if 'lga' in c.lower() and 'id' not in c.lower()), None)
        if lga_col:
            # Create regex for any selected LGA (escaped)
            # This allows "Gabasawa" to match "Gabasawa LGA"
            pattern = '|'.join([re.escape(str(x).strip()) for x in selected_lgas])
            df = df[df[lga_col].astype(str).str.contains(pattern, case=False, na=False)]
    
    # 2. Gender Filter
    if selected_genders:
        g_col = next((c for c in df.columns if 'gender' in c.lower()), None)
        if g_col:
            sel = [g.lower().strip() for g in selected_genders]
            df = df[df[g_col].astype(str).str.lower().str.strip().isin(sel)]

    # 3. Date Filter (Robust Parsing & Future Handling)
    if start_date and end_date and date_col and date_col in df.columns:
        # Try converting with format='mixed' which handles both ISO (YYYY-MM-DD) and Day-First (DD-MM-YYYY)
        dates = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
        
        # Smart Logic: If data contains future dates (like 2025) but filter ends today (2024),
        # extend the filter logic to include future data to prevent "No Data" errors in demo/projection datasets.
        data_max_date = dates.max()
        effective_end_date = end_date
        
        # Check if user-selected end_date is basically "today" (default) but data is in future
        import datetime
        if isinstance(end_date, datetime.date) and data_max_date is not pd.NaT:
             if end_date <= datetime.date.today() and data_max_date.date() > end_date:
                 # Auto-extend filter to accommodate future data points
                 effective_end_date = data_max_date.date()

        # Create mask
        mask = (dates.dt.date >= start_date) & (dates.dt.date <= effective_end_date)
        
        if include_na_dates:
            mask = mask | dates.isna()
            
        df = df[mask]

    return df