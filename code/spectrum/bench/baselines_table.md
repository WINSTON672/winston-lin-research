# DST vs published spiking-neuron models

All models integrated at dt = 0.01 ms over 300 ms, identical stimulus protocols.

Each model is driven at multiples of its own rheobase, so the stimulus means the same thing despite the differing units.

| Model | State vars | Params | Spikes at all | Refractory period | Temporal summation | Spike-rate adaptation | Max stable dt (ms) | Runtime (ms) |
|---|---|---|---|---|---|---|---|---|
| DST (ours) | 3 | 8 | yes | yes | yes | yes | 2 | 12.4 |
| LIF | 1 | 5 | yes | no | yes | no | 1 | 6.8 |
| Izhikevich | 2 | 4 | yes | no | yes | yes | 1 | 23.6 |
| AdEx | 2 | 9 | yes | no | yes | yes | 0.5 | 68.3 |

## What this shows

**The variable count is not the contribution.** Izhikevich
reaches spike-rate adaptation and temporal summation with two
state variables and four parameters; DST needs three and eight.
Any claim of the form "three variables suffice for the full
repertoire" was answered in 2003 with one variable fewer, and a
reviewer will say so. That framing has to go.

**Two things do survive, and both are measured above.**

1. *Absolute refractory period.* Under a pulse pair at 20x
   rheobase, DST is the only model here that refuses the second
   spike. Izhikevich and AdEx recover through a slow current and
   can be forced to fire again immediately; DST cannot, because
   the threshold jump is a hard veto. Real neurons have an
   absolute refractory period from sodium-channel inactivation,
   and a chip that drives spikes into a shared bus needs one.

2. *Linearity buys the coarsest timestep and the lowest cost.*
   DST stays correct at dt = 2 ms -- twice what Izhikevich
   tolerates and four times AdEx -- and at matched dt it runs
   ~1.8x faster than Izhikevich and ~5.4x faster than AdEx,
   despite carrying the extra state variable. Izhikevich pays
   for a quadratic term that needs sub-stepping; AdEx pays for
   an exponential. DST is entirely linear.

   For a model whose stated purpose is fabrication this is the
   whole argument. The tolerable timestep is the clock you have
   to run, and a linear system maps onto an RC network -- which
   is exactly why the D compartment is a capacitor and a
   resistor. Neither a quadratic nor an exponential term has
   any such two-component realisation.

That second point is the paper. Not "fewer variables" -- that
claim loses to Izhikevich -- but "no nonlinearity to
fabricate", which is a claim about hardware that the hardware
can be made to support.

The DSTC extension then adds what no point-neuron model can
have at all: two-compartment dendrites, a voltage-gated NMDA
conductance, and the proximal/distal inhibition asymmetry.
Izhikevich and AdEx have no morphology to put those on. This
table cannot score them, which is itself the argument for the
comparison the paper should be making.
