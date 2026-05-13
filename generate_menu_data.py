import pandas as pd
import json
import os

def generate():
    processed_path = 'data/processed/unified_car_data.csv'
    output_path = 'app/static/car_data.json'

    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} not found. Run training first!")
        return

    print("Reading dataset to extract brand and model mapping...")
    df = pd.read_csv(processed_path)

    df['Power_HP'] = (df['Power_kW'] / 0.735499).round().astype(int)

    menu_data = {}
    
    makes = sorted(df['Make'].dropna().unique())

    for make in makes:
        menu_data[make] = {}
        make_df = df[df['Make'] == make]
        
        models = sorted(make_df['Model'].dropna().unique())
        
        for model in models:
            hps = sorted(make_df[make_df['Model'] == model]['Power_HP'].unique().tolist())
            if hps:
                menu_data[make][model] = hps

    os.makedirs('app/static', exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(menu_data, f, indent=4)
    
    print(f"Success! Menu data saved to {output_path}")

if __name__ == "__main__":
    generate()