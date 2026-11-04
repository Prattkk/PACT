# Public code audit (verified excerpts), 2026-10-18

Shallow clones; findings apply only to these commits.

| Repo | Commit | Sharpe uses rf | Fill timing | Costs | Inference | Window |
|---|---|---|---|---|---|---|
| TauricResearch/TradingAgents | e8f5503 | n/a (no Sharpe; hit rate, mean alpha `tradingagents/backtest.py:195-206`) | next close (`memory/settlement.py:58-59`) | none ("not a portfolio simulator", `backtest.py:9-14`) | none | paper: 1 Jan-29 Mar 2024; paper script not released |
| pipiku915/FinMem-LLM-StockTrading | be814aa | rf = 0 (`data-pipeline/07-metrics.py:162`) | action at t earns close t->t+1 (`07-metrics.py:54`) | none | Wilcoxon on cumulative reward paths (`08-Wilcoxon-Test.py:28-31,61`) | 2022-10-06..2023-04-10 (`07-metrics.py:199-200`) |
| AI4Finance-Foundation/FinRobot | 2717499 | backtrader default 1% / desktop 4% | next bar | zero commission and slippage (`backtrader_adapter.py:191`) | none | n/a |
| virattt/ai-hedge-fund | 78b779c | none (`hedge_fund/backtesting/fund.py:197`) | fund: next close; PEAD engine: signal-day close | none (`brokers/sim.py:3-5`) | none | default 2024-06-01..today |
| waylonli/FINSABER | e72c262 | 3% (`trade_config.py:13`) | next open (`trade_config.py:17`) | per-order commission | none (means/medians) | 2004-2024 |

Verified by hand: FinMem rf=0 and close-t credit; ai-hedge-fund no-rf Sharpe and no-cost broker; FinRobot zero-commission warning.
