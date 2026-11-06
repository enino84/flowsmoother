"""F6. Headline: paired gain of the wake over the square, by lag.

Same seed, same network, same perturbed observations, same r, ridge and prior;
only the cross-time predecessors of the h rows differ.

  python f06_paired_gain_square.py --run pilot --dens 0.1
"""
from _common import parser
from _paired import figure

p = parser(__doc__)
p.add_argument("--dens", type=float, default=0.10)
p.add_argument("--N", type=int, default=None)
a = p.parse_args()
figure(a.run, "square", a.dens, a.N, "f06_paired_gain_square",
       "Gain of the wake over the square")
