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

def make_price_query_multi(pool_addresses, start_ts, end_ts):
    """Create a GraphQL query string for fetching price, volume, and liquidity for multiple pools."""
    pool_list = ', '.join(f'"{addr}"' for addr in pool_addresses)
    return f"""
{{
  poolHourDatas(
    first: 1000,
    where: {{
      pool_in: [ {pool_list} ]
      periodStartUnix_gte: {start_ts}
      periodStartUnix_lte: {end_ts}
    }},
    orderBy: periodStartUnix,
    orderDirection: asc
  ) {{
    pool
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

def fetch_pools_hourly_data_multi(api_key, subgraph_id, pool_addresses, start_date, end_date):
    """
    Fetch hourly price, volumeUSD, liquidity for multiple pools in one query.
    Returns a dict: pool_address -> DataFrame
    """
    start_ts = date_to_unix(start_date)
    end_ts = date_to_unix(end_date)
    batch_size = 20  # Number of pools per batch
    pool_dict = {addr: [] for addr in pool_addresses}
    for batch_start in range(0, len(pool_addresses), batch_size):
        batch = pool_addresses[batch_start:batch_start+batch_size]
        all_data = []
        last_ts = start_ts
        while True:
            graphql_query = make_price_query_multi(batch, last_ts, end_ts)
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
                print(f"[fetch_pools_hourly_data_multi] ERROR: Query failed for batch {batch} with status code {response.status_code}: {response.text}")
                break
            result = response.json()
            if 'data' not in result or 'poolHourDatas' not in result['data']:
                print(f"[fetch_pools_hourly_data_multi] WARNING: Missing 'data' or 'poolHourDatas' in response for batch {batch}. Response: {result}")
                break
            hour_data = result['data']['poolHourDatas']
            if not hour_data:
                break
            all_data.extend(hour_data)
            if len(hour_data) < 1000:
                break
            last_ts = int(hour_data[-1]['periodStartUnix']) + 1
        # Split by pool for this batch
        for row in all_data:
            pool_addr = row.get('pool')
            if pool_addr in pool_dict:
                pool_dict[pool_addr].append(row)
        print(f"[fetch_pools_hourly_data_multi] Fetched {len(all_data)} rows for batch {batch_start // batch_size + 1}/{(len(pool_addresses) + batch_size - 1) // batch_size}")
    # Convert to DataFrames
    for addr in pool_dict:
        pool_rows = pool_dict[addr]
        # Ensure pool_rows is a list; if not, set to empty list
        if not isinstance(pool_rows, list):
            pool_rows = []
        # If pool_rows is empty, create an empty DataFrame
        if not pool_rows:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(pool_rows)
            if not df.empty:
                df['periodStartUnix'] = pd.to_datetime(df['periodStartUnix'], unit='s')
                df = df.rename(columns={"token0Price": "price"})
                df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(float)
                df['volumeUSD'] = pd.to_numeric(df['volumeUSD'], errors='coerce').astype(float)
                df['liquidity'] = pd.to_numeric(df['liquidity'], errors='coerce').astype(float)
        pool_dict[addr] = df
    return pool_dict

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