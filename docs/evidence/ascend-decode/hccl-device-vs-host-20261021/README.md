# A collective costs 10 microseconds on the device; the rest is the host

The repository already held `tests/a3-calibration/hccl_latency.cpp`, a native
single-host HCCL calibration with no results page and no lesson citing it. It
builds with `build_hccl.sh` and runs with `--expansion aiv`. Sixteen devices,
AIV, 100 batches:

| case | bytes | ops a unit | device p50 | host wall | host release |
|---|---|---|---|---|---|
| independent_12kib | 12288 | 1 | **10.74** | 382.94 | 235.27 |
| chain_12kib | 12288 | 156 | 1564.14 | 5054.61 | 1783.35 |
| independent_48kib | 49152 | 1 | **15.21** | 355.22 | 211.14 |
| chain_48kib | 49152 | 156 | 2162.32 | 5564.41 | 2360.76 |

## Which overturns today's attribution

A single collective of 12 KB takes **10.74 us on the device** and 382.94 us of
host wall time. The 183 to 296 us measured through torch is therefore almost
entirely host-side issue and synchronisation, not a device handshake.

This project said the opposite earlier today: that the floor is "neither
bandwidth nor participants nor host dispatch", on the grounds that a kernel's
issue under graph replay is 1.064 us. That reasoning used the dispatch cost of
a *compute* kernel. A collective goes through a different host path, and the
native numbers separate them.

## And the chain shows the host cost amortises

156 collectives in a chain cost 1564.14 us on the device -- 10.0 us each,
matching the single call. But the host wall is 5054.61 for the chain, which is
**32.4 us a call** against 382.94 for one issued alone.

So the host overhead falls twelvefold when calls are issued back to back
without waiting between them.

## Which reverses the direction

The round's 374 us a layer of collective time is about two calls. On the device
that is 20 us. The rest is host path, and the host path is twelve times cheaper
per call when the calls are not serialised by a wait.

So the lever is not fewer calls -- it is **not waiting between them**. Whether
the engine's decode already issues its collectives back to back, or inserts a
synchronisation each time, is the measurement that follows, and it is a reading
of the engine rather than a sweep.
