"""Download the monthly series listed in DESIGN.md §1 into review/holdout_2025/raw/ (one CSV per series).

Run from the repository root:  python3 review/holdout_2025/fetch.py
Needs network access to query1/query2.finance.yahoo.com and fred.stlouisfed.org. Each file keeps the source's own
values; nothing is transformed here. A log of what was downloaded (first and last month, rows) goes to raw/_log.csv.
"""
import os, time, json, urllib.request, io
import pandas as pd

RAW = 'review/holdout_2025/raw'
os.makedirs(RAW, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (research script)'}

YAHOO = ['^GSPC', '^N225', '^SSMI', '^GDAXI', '^FTSE', '^GSPTSE', '^FCHI', 'FTSEMIB.MI', '^AEX', '^IBEX',
         'SPY', 'EWJ', 'EWL', 'EWG', 'EWU', 'EWC', 'EWQ', 'EWI', 'EWN', 'EWP',
         'JPY=X', 'CHF=X', 'EURUSD=X', 'GBPUSD=X', 'CAD=X', 'SEK=X', 'AUDUSD=X', 'NZDUSD=X',
         'ZN=F',
         'CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F', 'GC=F', 'SI=F', 'PL=F', 'PA=F', 'HG=F', 'ALI=F', 'ZC=F', 'ZS=F', 'ZW=F',
         'KE=F', 'ZL=F', 'ZM=F', 'ZO=F', 'KC=F', 'SB=F', 'CT=F', 'CC=F', 'OJ=F', 'LE=F', 'HE=F', 'GF=F']
FRED = ['TB3MS'] + [f'IRLTLT01{c}M156N' for c in ['US', 'JP', 'CH', 'DE', 'GB', 'CA', 'SE']] \
       + [f'IR3TIB01{c}M156N' for c in ['US', 'JP', 'CH', 'EZ', 'GB', 'CA', 'SE', 'AU', 'NZ']]

def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:                                   # network errors: back off 2, 4, 8 s
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))

def yahoo_monthly(t):
    """Month-end close and adjusted close (the bar dated the first of month M closes at the end of M)."""
    for host in ('query1', 'query2'):
        try:
            url = (f'https://{host}.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(t)}'
                   f'?period1=946684800&period2={int(time.time())}&interval=1mo&events=div,splits')
            res = json.loads(get(url))['chart']['result'][0]
            break
        except Exception:
            if host == 'query2':
                raise
    ts = pd.to_datetime(res['timestamp'], unit='s').to_period('M').to_timestamp('M')
    q = res['indicators']['quote'][0]
    adj = res['indicators'].get('adjclose', [{}])[0].get('adjclose', q['close'])
    df = pd.DataFrame({'date': ts, 'close': q['close'], 'adjclose': adj}).dropna(subset=['close'])
    return df.groupby('date').last().reset_index()

def fred_monthly(s):
    df = pd.read_csv(io.BytesIO(get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}')))
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df.date).dt.to_period('M').dt.to_timestamp('M')
    df['value'] = pd.to_numeric(df.value, errors='coerce')
    return df.dropna()

log = []
for t in YAHOO:
    try:
        df = yahoo_monthly(t); df.to_csv(f'{RAW}/yahoo_{t.replace("^", "IDX_").replace("=", "_")}.csv', index=False)
        log.append((t, 'yahoo', df.date.min().date(), df.date.max().date(), len(df), 'ok'))
    except Exception as e:
        log.append((t, 'yahoo', None, None, 0, f'failed: {e}'))
for s in FRED:
    try:
        df = fred_monthly(s); df.to_csv(f'{RAW}/fred_{s}.csv', index=False)
        log.append((s, 'fred', df.date.min().date(), df.date.max().date(), len(df), 'ok'))
    except Exception as e:
        log.append((s, 'fred', None, None, 0, f'failed: {e}'))
L = pd.DataFrame(log, columns=['series', 'source', 'first', 'last', 'rows', 'status'])
L.to_csv(f'{RAW}/_log.csv', index=False)
print(L.to_string(index=False))
print(f'\n{(L.status == "ok").sum()} of {len(L)} series downloaded')
