# The vendor operator costs 2.5x more inside a mixed process

2026-10-21, a3-84, one die. The same operator, the same shape -- 96 rows,
sixteen experts -- in two process states.

    clean process                        61.617 us   (event clock 61.036)
    after a domain build and a custom
    kernel has run                      153.325 us

Both clocks agree within one percent, so this is not a measurement artefact.

## Why it matters more than the number

The engine is a mixed process: it builds a communication domain before it
decodes. So the figure that belongs in the round's accounting is the high one,
and the recorded 157.2 us a layer stands -- it was measured in the right
state by accident.

But every vendor baseline taken in a clean probe understates what the engine
pays, and every one taken in a mixed probe overstates what a clean kernel
would. Two of today's comparisons sit on opposite sides of that: the
hand-written kernel's 2.33x disadvantage was measured against a vendor in the
same mixed state, so it is the honest comparison; the row sweep that showed
the operator is fixed cost was also mixed, so its conclusion holds.

## The rule

A baseline is a property of the process, not of the operator. Record which
state it was taken in, and compare only within one state.
