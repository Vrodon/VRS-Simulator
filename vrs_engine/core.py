"""
VRS Engine — core.py  [RETIRED]

This file has been superseded by the three-layer architecture:

    store.py      — Data overlord (holds matches_df + prizes_df)
    calculator.py — All mathematical operations applied to data
    pipeline.py   — Orchestrator (runs steps in order, returns standings)

The old compute_vrs() function is no longer used or exported.
Use run_vrs() from vrs_engine/__init__.py instead:

    from vrs_engine import Store, run_vrs

    store  = Store.from_valve(team_match_history, bo_prizes_map)
    result = run_vrs(store, cutoff=datetime.now())
"""
