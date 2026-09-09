# Method

## Scope

The Hunyuan3D 2.1 Paint experiments study appearance generation on fixed
geometry.
Every method receives the same fixed mesh and one selected conditioning view.
Inference represents that view with one input image. The protocol-corrected
training reader follows the upstream two-image conditioning contract by using
two distinct lighting renders from the same selected view. Shape generation is
outside the study, and remeshing is disabled. Holding geometry constant makes
differences in the final render easier to attribute to conditioning,
Paint-stage adaptation, or inference-time control.

The experiments use framed-panel assets because their appearance is
concentrated on a dominant front surface. Artwork and structural detail can be
assessed on that surface, while side and rear surfaces provide distinct
non-front regions for studying unintended content transfer.

## Base model and Corrected-Conditioning Baseline

Two terms have deliberately different meanings:

- **Base model** means the official Hunyuan3D-Paint weights, without a training
  update introduced by this project.
- **Corrected-Conditioning Baseline** means the complete experimental
  configuration: Base model, fixed mesh, selected conditioning image,
  inference settings, and fixed-view evaluation protocol.

The baseline is therefore not another name for the weights. It is the common
starting point against which all Paint-stage interventions are evaluated.

A preliminary diagnostic found a consistent association between
front-informative conditioning and higher rerendering fidelity. Conditioning
view 005 was consequently fixed as the shared input rule for the subsequent
controlled comparisons.

## Three view roles

The pipeline contains three view systems that must not be conflated:

1. The **conditioning view** is the camera associated with the input image.
2. A **generation-view slot** is one of the internal cameras used by
   Hunyuan3D-Paint while generating appearance.
3. An **evaluation view** is an external renderer camera used only after the
   textured 3D asset has been produced.

Matching numeric indices do not imply matching roles. An image may also serve
more than one purpose—for example, as the conditioning image and as the
ground-truth reference for one evaluation view—without merging the underlying
camera stages.

## Small-data Paint-stage adaptation

The adaptation study compares complete configurations rather than treating all
differences as isolated ablations.

### Broad-Scope Fine-Tuning

Broad-Scope Fine-Tuning starts from the official PBR initialisation and updates
the broad partial-network scope defined by the upstream training setup. Early
experiments use 32 and 80 training assets, with 500 updates in each case, to
test broader adaptation at two levels of data coverage.

### Reference-Conditioning LoRA

Reference-Conditioning LoRA freezes the Base model and inserts rank-4 low-rank
adapters into the query, key, value, and output projections of Reference
Attention and the DINO-derived conditioning pathway. One adapter is trained for
300 updates at a constant learning rate of `5e-5`; inference evaluates adapter
scales 0.50, 0.75, and 1.00.

### Protocol-corrected configurations

The corrected training reader better matches the front-informative input used
at inference. It samples the selected conditioning view with probability 0.50,
an alternate front-oblique view with probability 0.30, and each of the other
four views with probability 0.05. For that selected view, an ordered pair of
two distinct lighting conditions is sampled uniformly without replacement from
AL, ENVMAP, and PL. The frozen protocol uses base and schedule seeds of 42;
dataset decisions derive reproducible seeds from rank, worker, epoch, and asset
identity. Target ordering is fixed and spatial augmentation is disabled. The
training wrapper uses conditioning-dropout probability 0.10 and deterministically
retries a step seed, up to 128 times, when necessary to keep the Multi-View
Attention pathway active.

Two methods share this protocol, the same 80-asset split, a fresh official PBR
initialisation, the same example sequence, and a 320-update budget:

- **Protocol-Corrected Broad-Scope Fine-Tuning** updates the broad
  partial-network scope. Its peak learning rate is `5e-7`, and the evaluated
  checkpoint is step 320.
- **Protocol-Corrected Multi-View-Attention Fine-Tuning** updates only the
  query, key, value, and output projections in the multi-view-attention blocks.
  Its peak learning rate is `1e-6`, and the evaluated checkpoint is step 160.

The instantiated Paint UNet contains approximately 1.047 billion trainable
parameters (53.35%) in the Broad configuration and 49.6 million (2.53%) in the
MVA configuration. The methods differ in update capacity, learning rate, and
selected checkpoint, so conclusions concern each complete configuration.

## View-Selective Conditioning Gating

View-Selective Conditioning Gating is an inference-time intervention. It keeps
the official Base model weights fixed and modulates the completed outputs of
Reference Attention and DINO conditioning before those outputs are added to
the parent residual.

For generation-view slot \(v\) and direct-conditioning pathway
\(p\in\{\mathrm{ref},\mathrm{dino}\}\), the operator is

\[
\widetilde{R}^{(p)}_v = m_v R^{(p)}_v, \qquad m_v\in\{0,1\}.
\]

The binary gate is derived from live generation-camera metadata. If
\(\widehat{\mathbf d}_v\) is the unit viewing direction for slot \(v\) and
\(\widehat{\mathbf f}\) is the canonical front direction, then

\[
\theta_v = \frac{180}{\pi}\arccos\!\left(
\operatorname{clip}(\widehat{\mathbf d}_v^{\mathsf T}
\widehat{\mathbf f},-1,1)\right),
\qquad
m_v = \begin{cases}
1, & \theta_v < \tau,\\
0, & \theta_v \geq \tau.
\end{cases}
\]

For the official six internal slots—Front, Right, Back, Left, Top, Bottom—and
\(\tau=120^\circ\), the mask is `[1, 1, 0, 1, 1, 1]`. Only the rear-facing
slot is suppressed. Reference Attention and DINO conditioning share the mask;
Multi-View Attention is not gated directly. It can still receive an indirect
effect because it operates on hidden states already modified by the two gated
branches.

Before activation, the runtime verifies module identity, camera ordering, and
tensor layout. A mismatch stops the run before generation. After inference,
the original runtime state is restored, and every comparison variant begins
from a separately initialised inference context. The implementation lives in
`hy3dft.view_selective_gating` and exposes
`ViewSelectiveConditioningGate`.

## Evaluation objectives

Generated assets are rerendered from a predefined six-view orthographic camera
set and compared with corresponding ground-truth renders. Results are reported
separately for front views, the selected input view, non-front views, and all
views.

Pixel metrics provide repeatable evidence but do not by themselves prove that
hidden-surface leakage has been resolved. The review therefore also checks
whether recognisable front-visible content appears on side or rear surfaces and
whether a method introduces new artefacts.
