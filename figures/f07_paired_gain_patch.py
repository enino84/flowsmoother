"""F7. Paired gain of the wake over the size control (isotropic patch with the
same number of predecessors): position against quantity.

  python f07_paired_gain_patch.py --run pilot --dens 0.1
"""
from _common import parser
from _paired import figure

p = parser(__doc__)
p.add_argument("--dens", type=float, default=0.10)
p.add_argument("--N", type=int, default=None)
a = p.parse_args()
figure(a.run, "patch", a.dens, a.N, "f07_paired_gain_patch",
       "Gain of the wake over an isotropic patch of the same size")
