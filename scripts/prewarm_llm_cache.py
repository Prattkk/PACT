"""Fill the LLM response cache for one cell configuration in parallel.

`protocol.run(as_of)` depends only on the decision date (portfolio state is
applied afterwards by the engine), so the weekly calls are independent. This
script runs them with a thread pool; the normal matrix / ablation / LOO
scripts then replay every response from the cache, so their results are
exactly what a sequential run would produce. A prompt that differs from the
pre-filled one is simply a cache miss and is fetched during the normal run.

Example:
    python -m scripts.prewarm_llm_cache --protocol sequential_pipeline --altdata --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from backtest.engine import WARMUP_CALENDAR_DAYS, decision_dates
from data.fetchers.prices import fetch_panel
from data.universe import TRADEABLE_SYMBOLS
from pact_logging import get_logger
from scripts._common import build_agents, build_llm_client, build_protocol, load_config

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--window", default="b")
    p.add_argument("--regime", default="frontier")
    p.add_argument("--altdata", action="store_true")
    p.add_argument("--secondary", nargs="*", default=[])
    p.add_argument("--null-agent", default=None, help="replace this agent with a NullAgent (LOO)")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    cfg = load_config("ablation_matrix.yaml")
    blk = cfg[f"window_{a.window}"]
    start = datetime.strptime(blk["start"], "%Y-%m-%d").date()
    end = datetime.strptime(blk["end"], "%Y-%m-%d").date()
    universe = TRADEABLE_SYMBOLS

    llm = build_llm_client(a.regime)

    def build():
        agents = build_agents(llm, universe, cell_window=(start, end),
                              enable_altdata=a.altdata, enable_secondary=tuple(a.secondary))
        if a.null_agent:
            from agents.null_agent import NullAgent

            agents[a.null_agent] = NullAgent(replaces_name=a.null_agent, universe=universe)
        return build_protocol(a.protocol, agents, llm, universe)

    prices = fetch_panel(list(universe), start - timedelta(days=WARMUP_CALENDAR_DAYS), end + timedelta(days=1),
                         field="Adj Close")
    dates = [d.date() for d in decision_dates(prices.index, start, end, cfg["backtest"]["rebalance_freq"])]
    log.info("prewarm protocol=%s weeks=%d workers=%d", a.protocol, len(dates), a.workers)

    # Some protocols pass inputs to agents through setters (PM, cross-asset),
    # so each worker gets its own agent set; first calls run sequentially so
    # every instance loads its data panels before the threads start.
    protos = [build() for _ in range(a.workers)]
    for pr in protos:
        pr.run(dates[0])
    chunks = [dates[1:][i::a.workers] for i in range(a.workers)]
    done = 1

    def work(pr, ds):
        for d in ds:
            pr.run(d)
        return len(ds)

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, pr, ds) for pr, ds in zip(protos, chunks)]
        for f in as_completed(futs):
            done += f.result()
            print(f"{a.protocol} {done}/{len(dates)}", flush=True)
    print(f"{a.protocol} done {done}/{len(dates)}", flush=True)


if __name__ == "__main__":
    main()
