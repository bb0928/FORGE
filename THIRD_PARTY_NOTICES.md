# Third-party provenance

| Component | Source | License / bundled notice |
|---|---|---|
| `rl/std_l2_ppo.py` | [RSL-RL v5.2.0](https://github.com/leggedrobotics/rsl_rl/tree/v5.2.0), with FORGE StdL2 additions | BSD-3-Clause; `LICENSES/RSL-RL-BSD-3-Clause.txt` and retained file header |
| G1 base XML and meshes | [mjlab v1.4.0](https://github.com/mujocolab/mjlab/tree/v1.4.0), Unitree G1 resources | mjlab Apache-2.0 and original robot attribution; `LICENSES/MJLAB-Apache-2.0.txt`, `LICENSES/Unitree-BSD-3-Clause.txt` |
| Dex3-1 hand geometry and inertias | [unitree_ros commit d96d8f6](https://github.com/unitreerobotics/unitree_ros/tree/d96d8f63ae17a7108d4f7229c00ef875ba7129c9/robots/dexterous_hand_description/dex3_1) | BSD-3-Clause; `LICENSES/Unitree-BSD-3-Clause.txt` |

The historical asset note identifies the Unitree hand source above. Finger joints
are fixed at their zero pose; payload bodies were added for the milestone. The
unmodified G1 source XML SHA256 is
`febdcbeffbbf84051556ae41a5ac1b43fb479a5d76bdb3f54824dbc2721c20aa`.
The packaged modified XML's original/release hashes are in `SOURCE_MANIFEST.json`.

Runtime packages such as PyTorch, MuJoCo, Warp, Viser and mjlab are installed as
dependencies, not bundled into this source archive. Their own distributions carry
their license terms.
