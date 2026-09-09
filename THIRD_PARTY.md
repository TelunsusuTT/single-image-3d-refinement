# Third-Party Components

## Hunyuan3D 2.1

This project integrates with the official Hunyuan3D 2.1 Paint-stage source and
model weights. They are external dependencies and are not redistributed in this
repository. Their use is governed by the upstream project and model licences.

Set `HUNYUAN3D_ROOT` to an external Hunyuan3D 2.1 checkout. If its Paint
source is not located at `hy3dpaint/` within that checkout, also set
`HUNYUAN3D_PAINT_SOURCE_ROOT`. Official weight placement follows the upstream
Paint configuration and installation instructions.

## TRELLIS.2

The `trellis2_mv_adapter/` pipeline integrates with the official
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) source and model weights.
They remain external to this repository and are governed by the upstream
licence and model terms.

## MV-Adapter

Multi-view image generation and texture baking use the official
[MV-Adapter](https://github.com/huanngzh/MV-Adapter) implementation and model
weights. The upstream source, base diffusion models, and weights are not
redistributed here and remain subject to their respective terms.

## Amazon Berkeley Objects

The framed-panel study subset is derived from Amazon Berkeley Objects (ABO).
This repository versions portable asset identifiers and split metadata only;
source meshes, images, and generated training examples remain external. Access
and redistribution are subject to the ABO terms.

## Blender

Blender is used as an external executable for isolated OBJ-to-GLB conversion
and fixed-view rerendering. Blender is not vendored by this repository.

## Responsibility for external material

This repository does not grant rights to third-party code, weights, data, or
assets. Anyone reproducing the experiments is responsible for obtaining those
materials and complying with their respective licences and terms.
