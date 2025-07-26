import requests

import pandas as pd

from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed

def date_to_unix(date_str):
    """
    Convert a date string in 'YYYY-MM-DD' format to a Unix timestamp (seconds since epoch).
    Args:
        date_str (str): Date string in 'YYYY-MM-DD' format.
    Returns:
        int: Unix timestamp.
    """
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return int(dt.timestamp())

def make_price_query(pool_address, start_ts, end_ts):
    """
    Create a GraphQL query string for fetching hourly price, volumeUSD, and liquidity for a single Uniswap V3 pool.
    Args:
        pool_address (str): Pool address.
        start_ts (int): Start timestamp (Unix).
        end_ts (int): End timestamp (Unix).
    Returns:
        str: GraphQL query string.
    """
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
    """
    Create a GraphQL query string for fetching hourly price, volumeUSD, and liquidity for multiple Uniswap V3 pools.
    Args:
        pool_addresses (list): List of pool addresses.
        start_ts (int): Start timestamp (Unix).
        end_ts (int): End timestamp (Unix).
    Returns:
        str: GraphQL query string.
    """
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
    pool {{
      id
    }}
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
    Args:
        api_key (str): The Graph API key.
        subgraph_id (str): Subgraph ID.
        pool_address (str): Pool address.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame with columns periodStartUnix, price, volumeUSD, liquidity.
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

def fetch_pools_hourly_data_batched(api_key, subgraph_id, pool_addresses, start_date, end_date):
    """
    Fetch hourly price, volumeUSD, and liquidity for multiple pools in sequential batches.

    Args:
        api_key (str): The Graph API key.
        subgraph_id (str): Subgraph ID.
        pool_addresses (list): List of pool addresses (lowercase strings).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        dict: Mapping pool_address -> DataFrame of hourly data.
    """
    start_ts = date_to_unix(start_date)
    end_ts = date_to_unix(end_date)
    batch_size = 10  # Number of pools per batch
    pool_dict = {addr: [] for addr in pool_addresses}

    for batch_start in range(0, len(pool_addresses), batch_size):
        batch = pool_addresses[batch_start:batch_start + batch_size]
        all_data = []
        last_ts = start_ts

        while True:
            graphql_query = make_price_query_multi(batch, last_ts, end_ts)
            payload = {
                "query": graphql_query
            }
            url = f"https://gateway.thegraph.com/api/subgraphs/id/{subgraph_id}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"[ERROR] Query failed for batch {batch} with status code {response.status_code}: {response.text}")
                break

            result = response.json()
            hour_data = result.get("data", {}).get("poolHourDatas", [])
            if not hour_data:
                break

            all_data.extend(hour_data)
            if len(hour_data) < 1000:
                break
            last_ts = int(hour_data[-1]['periodStartUnix']) + 1

        # Group results by pool address
        for row in all_data:
            pool_obj = row.get('pool', {})
            addr = pool_obj.get('id')
            if not addr:
                # print(f"[WARNING] Missing pool ID in row: {row}")
                continue
            if addr not in pool_dict:
                # print(f"[WARNING] Pool address {addr} not in pool_dict (skipping)")
                continue
            pool_dict[addr].append(row)

        print(f"[fetch_pools_hourly_data_batched] Fetched {len(all_data)} rows for batch {batch_start // batch_size + 1}/{(len(pool_addresses) + batch_size - 1) // batch_size}")

    # Convert to DataFrames
    for addr, rows in pool_dict.items():
        if not isinstance(rows, list) or not rows:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(rows)
            if not df.empty:
                df['periodStartUnix'] = pd.to_datetime(df['periodStartUnix'], unit='s')
                df = df.rename(columns={"token0Price": "price"})
                df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(float)
                df['volumeUSD'] = pd.to_numeric(df['volumeUSD'], errors='coerce').astype(float)
                df['liquidity'] = pd.to_numeric(df['liquidity'], errors='coerce').astype(float)
        pool_dict[addr] = df

    return pool_dict

def fetch_pools_hourly_data_batched_parallel(api_key, subgraph_id, pool_addresses, start_date, end_date, max_workers=8):
    """
    Fetch hourly price, volumeUSD, and liquidity for multiple pools using asynchronous batch requests.

    Args:
        api_key (str): The Graph API key.
        subgraph_id (str): Subgraph ID.
        pool_addresses (list): List of pool addresses (lowercase strings).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        max_workers (int): Maximum number of threads for async requests.

    Returns:
        dict: Mapping pool_address -> DataFrame of hourly data.
    """
    start_ts = date_to_unix(start_date)
    end_ts = date_to_unix(end_date)
    batch_size = 10
    pool_dict = {addr: [] for addr in pool_addresses}

    def fetch_batch(batch, batch_idx):
        batch_data = []
        last_ts = start_ts
        while True:
            graphql_query = make_price_query_multi(batch, last_ts, end_ts)
            payload = {
                "query": graphql_query
            }
            url = f"https://gateway.thegraph.com/api/subgraphs/id/{subgraph_id}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"[ERROR] Batch {batch} failed: {response.status_code} {response.text}")
                return []
            result = response.json()
            hour_data = result.get("data", {}).get("poolHourDatas", [])
            if not hour_data:
                break
            batch_data.extend(hour_data)
            if len(hour_data) < 1000:
                break
            last_ts = int(hour_data[-1]["periodStartUnix"]) + 1
        print(f"[fetch_pools_hourly_data_multi] Fetched {len(batch_data)} rows for batch {batch_idx+1}/{(len(pool_addresses) + batch_size - 1) // batch_size}")
        return batch_data

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, len(pool_addresses), batch_size):
            batch = pool_addresses[batch_start:batch_start + batch_size]
            batch_idx = batch_start // batch_size
            futures.append(executor.submit(fetch_batch, batch, batch_idx))
        for fut in as_completed(futures):
            batch_result = fut.result()
            # print(f"[fetch_pools_hourly_data_multi] Received batch result with {len(batch_result)} rows")
            for row in batch_result:
                pool_obj = row.get("pool", {})
                addr = pool_obj.get("id")
                if not addr:
                    # print(f"[fetch_pools_hourly_data_multi] WARNING: Missing pool ID in row: {row}")
                    continue
                if addr not in pool_dict:
                    # print(f"[fetch_pools_hourly_data_multi] WARNING: Pool address {addr} not in pool_dict")
                    continue
                pool_dict[addr].append(row)

    # Convert to DataFrames
    # print("[fetch_pools_hourly_data_multi] Converting results to DataFrames...")
    for addr, rows in pool_dict.items():
        df = pd.DataFrame(rows)
        # print(f"[fetch_pools_hourly_data_multi] Pool {addr} has {len(df)} rows (empty={df.empty})")
        if not df.empty:
            df['periodStartUnix'] = pd.to_datetime(df['periodStartUnix'], unit='s')
            df = df.rename(columns={"token0Price": "price"})
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['volumeUSD'] = pd.to_numeric(df['volumeUSD'], errors='coerce')
            df['liquidity'] = pd.to_numeric(df['liquidity'], errors='coerce')
        pool_dict[addr] = df

    return pool_dict


def fetch_all_pool_addresses(api_key, subgraph_id, pool_tier=None, first=1000):
    """
    Fetch all pool addresses from a Uniswap subgraph, with optional filtering by pool tier.
    Args:
        api_key (str): The Graph API key.
        subgraph_id (str): Subgraph ID.
        pool_tier (str, optional): Pool tier ('LOW', 'MEDIUM', 'HIGH') or None for no filter.
        first (int): Number of pools to fetch per request.
    Returns:
        list: List of pool addresses (str).
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
        if 'data' not in result or 'pools' not in result['data']:
            raise Exception(f"Unexpected response format: {result}")
        pools = result['data']['pools']
        if not pools:
            break
        pool_addresses.extend([p['id'] for p in pools])
        if len(pools) < first:
            break
        skip += first
    return pool_addresses