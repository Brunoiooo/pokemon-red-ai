"""Hand-verified parent -> child event edges for whatever tools/gen_event_
graph.py's automatic passes (2: intra-function control-flow analysis, 5:
per-map dispatch-table state order) still can't see -- a `call` into a
shared helper that itself branches, a condition tested via something other
than CheckEvent/wYCoord/wXCoord, or any other idiom neither pass models.

Empty by default. When a human spots a real gap while reading a script
(tools/view_script_graph.py's graph viewer is built for exactly this),
confirm the dependency by eye and add it here. Merged into EVENT_GRAPH at
generation time (see gen_event_graph.py main()), so it goes through the
same cycle-detection/blacklist safety net as every auto-inferred edge -- a
wrong manual entry is still caught, not silently trusted either.

NOT regenerated -- edit by hand.

Format: child_event -> frozenset of parent_events.

The idiom neither pass models yet, all six entries below: `ld b, ITEM_X /
call IsItemInBag / jr z|nz, ...` gating a SetEvent -- an inventory check,
not a CheckEvent, so it's invisible to both the CFG guard analysis and the
map-script state-guard chain. Confirmed by reading every scripts/*.asm
IsItemInBag call site (9 total): the other 3 (CinnabarIsland's locked-door
idempotency check, RocketHideoutElevator's/Route16Gate1F's plain access
gates) don't gate a SetEvent at all, and CopycatsHouse2F's TM31-for-a-
Poke-Doll trade checks a plain purchasable item with no corresponding
"got it" event -- nothing to link to for those.
"""
from __future__ import annotations

MANUAL_PARENT_EDGES: dict[str, frozenset[str]] = {
    # OaksLab.asm: OaksLabOakGivesPokedexScript only runs once
    # `ld b, OAKS_PARCEL / call IsItemInBag` succeeds (talking to Oak with
    # the parcel in hand) -- both events it sets share this prerequisite.
    "EVENT_OAK_GOT_PARCEL": frozenset({"EVENT_GOT_OAKS_PARCEL"}),
    "EVENT_GOT_POKEDEX": frozenset({"EVENT_GOT_OAKS_PARCEL"}),
    # BikeShop.asm: the clerk only exchanges a bicycle for a Bike Voucher
    # already in the bag.
    "EVENT_GOT_BICYCLE": frozenset({"EVENT_GOT_BIKE_VOUCHER"}),
    # GameCorner.asm: all three NPCs handing out coins require the Coin
    # Case (from CeladonDiner) to already be in the bag.
    "EVENT_GOT_10_COINS": frozenset({"EVENT_GOT_COIN_CASE"}),
    "EVENT_GOT_20_COINS": frozenset({"EVENT_GOT_COIN_CASE"}),
    "EVENT_GOT_20_COINS_2": frozenset({"EVENT_GOT_COIN_CASE"}),
}
