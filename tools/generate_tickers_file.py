import json
import os

# Base manual list for things not in SEC list (Indices, Crypto)
MANUAL_ENTRIES = {
    # --- Indices ---
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "Dow Jones Industrial Average (^DJI)": "^DJI",
    "Russell 2000 (^RUT)": "^RUT",
    "VIX Volatility Index (^VIX)": "^VIX",
    # --- Crypto ---
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD",
}

def generate_tickers():
    json_path = 'data/company_tickers.json'
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Start with manual entries
    tickers_map = MANUAL_ENTRIES.copy()
    
    # Process SEC data
    for key, value in data.items():
        ticker = value['ticker']
        title = value['title']
        
        # Simple heuristic for nicer display:
        if title.isupper():
            title = title.title()
            
        display_name = f"{title} ({ticker})"
        tickers_map[display_name] = ticker

    # Sort the map by Display Name for the dropdown
    sorted_items = sorted(tickers_map.items(), key=lambda x: x[0])
    
    # Write to tickers.py
    with open('tickers.py', 'w', encoding='utf-8') as f:
        f.write('# Auto-generated file containing popular tickers\n')
        f.write('# Generated from SEC company_tickers.json and manual entries\n\n')
        f.write('POPULAR_TICKERS_MAP = {\n')
        
        # Write entries
        for display, ticker in sorted_items:
            # Use repr() to automatically handle escaping of quotes and backslashes
            f.write(f"    {repr(display)}: '{ticker}',\n")
            
        f.write('}\n\n')
        f.write('POPULAR_TICKERS_LIST = list(POPULAR_TICKERS_MAP.keys())\n')
    
    print(f"Successfully generated tickers.py with {len(sorted_items)} entries.")

if __name__ == "__main__":
    generate_tickers()
