"""
Synthetic-market Monte Carlo for the backtest engine.

This package plugs into the engine's already-injected data seams (``fetch_fn`` and
``chain_fetch_fn`` in ``engine.runner.run_backtest`` / ``engine.data_store.DataProvider``)
to drive a backtest off price paths synthesized from the Bates model (Heston stochastic
volatility + Merton-style log-normal jumps) rather than real ThetaData.

                bates.py           option_pricer.py             market_maker.py
    RNG seed -> spot+vol path -> (S_t, V_t, K, T, r) -> bid/ask/greeks -> run_backtest(fetch_fn=...)
                              ^                                   ^
                    bias V_t into an IV surface          fills at bid/ask; no slippage in engine

For intraday 0DTE XSP (cash-settled, European-style, ~4:00pm PM settlement) the strategy
only ever reads today's expiring chain and today's expiring contracts' bid/ask/greeks/
``underlying_price``, so each Monte Carlo path synthesizes ONE trading session rather than a
long multi-day history. See ``engine/greeks.py`` and ``engine/thetadata_client.py`` for the
column/history conventions this package must stay consistent with.
"""
