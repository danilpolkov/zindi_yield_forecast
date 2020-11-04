import numpy as np
import pandas as pd




def calc_bands(fid, monthes, bands_of_interest, band_names, folder='../data/raw/image_arrays_train'):
    fn = f'{folder}/{fid}.npy'
    arr = np.load(fn)

    values = {}
    for month in monthes:
        bns = [str(month) + '_' + b for b in bands_of_interest] # Bands of interest for this month 
        idxs = np.where(np.isin(band_names, bns)) # Index of these bands
        vs = arr[idxs, 20, 20] # Sample the im at the center point
        for bn, v in zip(bns, vs[0]):
            values[bn] = v
    return values


def calc_vegetation_indexes(df):
    monthes = set(col.split('_')[0] for col in df.columns if col not in ['Field_ID', 'Yield'])
    #print(monthes)
    for month in monthes:
        # ndvi
        tru_month = int(month)
        ndvi = (df[f'{month}_S2_B8'] - df[f'{month}_S2_B4']) / (df[f'{month}_S2_B8'] + df[f'{month}_S2_B4']) 
        df[f'{tru_month}_ndvi'] = ndvi
        #evi
        evi = 2.5 * (df[f'{month}_S2_B8'] - df[f'{month}_S2_B4']) / \
                             (df[f'{month}_S2_B8'] + 6 * df[f'{month}_S2_B4'] - 7.5 * df[f'{month}_S2_B2'] + 1) 

        df[f'{tru_month}_evi'] = evi
        
        grvi = df[f'{month}_S2_B8'] / df[f'{month}_S2_B3']
        df[f'{tru_month}_grvi'] = grvi
        
        gndvi = (df[f'{month}_S2_B8'] - df[f'{month}_S2_B3']) / (df[f'{month}_S2_B8'] + df[f'{month}_S2_B3'])
        df[f'{tru_month}_gndvi'] = gndvi
        
        alpha = 0.1
        wrdvi = (alpha*df[f'{month}_S2_B8'] - df[f'{month}_S2_B4']) / (alpha*df[f'{month}_S2_B8'] + df[f'{month}_S2_B4']) 
        df[f'{tru_month}_wrdvi'] = wrdvi
    return df


def process_dataset(df, monthes, bands_of_interest, band_names, folder, drop_s2_columns = True):
    df = pd.DataFrame([calc_bands(fid, monthes, bands_of_interest, band_names, folder) for fid in df['Field_ID'].values])
    df_with_indexes = calc_vegetation_indexes(df)
    if drop_s2_columns:
        drop_s2_columns = [col for col in df_with_indexes.columns if col[-2] == 'B']
        df_with_indexes = df_with_indexes.drop(drop_s2_columns, 1)
    
    # ww inf and nan
    df_with_indexes = df_with_indexes.replace([np.inf, -np.inf], np.nan)
    df_with_indexes = df_with_indexes.fillna(0)
    print(df_with_indexes.shape)
    return df_with_indexes