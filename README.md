# DeepMarket

ARTE 4110U final project, Team NCS. Does adding broad-market context (SPY/QQQ) help an LSTM predict whether a stock's next-day close is UP or NOT-UP?

```
Stock OHLCV + SPY/QQQ context -> 30-day sequence -> PyTorch LSTM -> P(next-day UP)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All experiment settings live in `configs/default.yaml`.

*Work in progress: pipeline code is being added.*
