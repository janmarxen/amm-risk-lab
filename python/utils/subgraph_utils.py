import requests

import pandas as pd

from datetime import datetime

def date_to_unix(date_str):
    """Convert a date string in 'YYYY-MM-DD' format to a Unix timestamp."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return int(dt.timestamp())

def make_price_query(pool_address, start_ts, end_ts):
    """Create a GraphQL query string for fetching price, volume, and liquidity of a Uniswap V3 pool."""
    return f"""
    {{
      poolHourDatas(first: 1000, where: {{pool: \"{pool_address}\", periodStartUnix_gte: {start_ts}, periodStartUnix_lte: {end_ts}}}, orderBy: periodStartUnix, orderDirection: asc) {{
        periodStartUnix
        token0Price
        volumeUSD
        liquidity
      }}
    }}
    """

def fetch_pool_hourly_data(api_key, subgraph_id, pool_address, start_date, end_date):
    """
    Fetch hourly price (token0Price as 'price'), volumeUSD, liquidity, and datetimes for a Uniswap V3 pool from the subgraph.
    Dates should be in 'YYYY-MM-DD' format.
    Returns a pandas DataFrame with columns: periodStartUnix, price, volumeUSD, liquidity
    """
    start_ts = date_to_unix(start_date)
    end_ts = date_to_unix(end_date)
    all_data = []
    last_ts = start_ts
    while True:
        graphql_query = make_price_query(pool_address, last_ts, end_ts)
        payload = {
            "query": graphql_query,
            "operationName": "Subgraphs",
            "variables": {}
        }
        url = f"https://gateway.thegraph.com/api/subgraphs/id/{subgraph_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Query failed with status code {response.status_code}: {response.text}")
        result = response.json()
        hour_data = result['data']['poolHourDatas']
        if not hour_data:
            break
        all_data.extend(hour_data)
        if len(hour_data) < 1000:
            break
        # Update last_ts to one after the last returned timestamp to avoid overlap
        last_ts = int(hour_data[-1]['periodStartUnix']) + 1

    df = pd.DataFrame(all_data)
    if not df.empty:
        df['periodStartUnix'] = pd.to_datetime(df['periodStartUnix'], unit='s')
        df = df.rename(columns={"token0Price": "price"})
        df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(float)
        df['volumeUSD'] = pd.to_numeric(df['volumeUSD'], errors='coerce').astype(float)
        df['liquidity'] = pd.to_numeric(df['liquidity'], errors='coerce').astype(float)
    return df

def fetch_all_pool_addresses(api_key, subgraph_id, pool_tier=None, first=1000):
    """
    Fetch all pool addresses from a Uniswap subgraph with optional filtering by pool_tier.
    - pool_tier: e.g., 'LOW', 'MEDIUM', 'HIGH', or None (no filter)
    Returns a list of pool addresses.
    """
    pool_addresses = []
    skip = 0
    while True:
        filters = []
        if pool_tier:
            filters.append(f'tier: "{pool_tier.upper()}"')
        where_clause = (', '.join(filters))
        where_str = f'where: {{{where_clause}}}' if where_clause else ''
        query = f"""
        {{
          pools(first: {first}, skip: {skip} {',' if where_str else ''}{where_str}) {{
            id
          }}
        }}
        """
        payload = {
            "query": query,
            "operationName": "Subgraphs",
            "variables": {}
        }
        url = f"https://gateway.thegraph.com/api/subgraphs/id/{subgraph_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Query failed with status code {response.status_code}: {response.text}")
        result = response.json()
        pools = result['data']['pools']
        if not pools:
            break
        pool_addresses.extend([p['id'] for p in pools])
        if len(pools) < first:
            break
        skip += first
    return pool_addresses