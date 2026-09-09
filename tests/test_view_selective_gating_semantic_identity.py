from __future__ import annotations

import json
import struct
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, PngImagePlugin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.view_selective_gating.semantic_identity import compare_artifact_semantics


def png_bytes(pixel: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), pixel).save(buffer, format="PNG")
    return buffer.getvalue()


def write_glb(
    path: Path,
    *,
    object_name: str,
    material_name: str,
    geometry_delta: float = 0.0,
    base_color_texture_index: int = 0,
) -> None:
    positions = struct.pack(
        "<9f",
        0.0 + geometry_delta,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    uvs = struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    indices = struct.pack("<3H", 0, 1, 2)
    embedded = png_bytes((20, 40, 60))
    binary = positions + uvs + indices + b"\0\0" + embedded
    binary += b"\0" * ((-len(binary)) % 4)
    image_offset = len(positions) + len(uvs) + len(indices) + 2
    document = {
        "asset": {"version": "2.0", "generator": "test-exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": object_name, "mesh": 0}],
        "meshes": [
            {
                "name": object_name,
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": material_name,
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": base_color_texture_index},
                    "metallicRoughnessTexture": {"index": 1},
                    "metallicFactor": 0.7,
                    "roughnessFactor": 0.4,
                },
            }
        ],
        "textures": [{"source": 0}, {"source": 0}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(uvs),
            },
            {
                "buffer": 0,
                "byteOffset": len(positions) + len(uvs),
                "byteLength": len(indices),
            },
            {
                "buffer": 0,
                "byteOffset": image_offset,
                "byteLength": len(embedded),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 3,
                "type": "VEC2",
            },
            {
                "bufferView": 2,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ],
    }
    json_payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    total = 12 + 8 + len(json_payload) + 8 + len(binary)
    payload = (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_payload), 0x4E4F534A)
        + json_payload
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_texture(
    path: Path,
    pixel: tuple[int, int, int],
    *,
    metadata: str,
) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("container-note", metadata)
    Image.new("RGB", (3, 2), pixel).save(path, format="PNG", pnginfo=info)


def make_package(
    root: Path,
    *,
    object_name: str,
    material_name: str,
    metadata: str,
    albedo_pixel: tuple[int, int, int] = (10, 20, 30),
    geometry_delta: float = 0.0,
    material_assignment: str = "textured_mesh.jpg",
    base_color_texture_index: int = 0,
) -> Path:
    root.mkdir(parents=True)
    write_glb(
        root / "textured_mesh.glb",
        object_name=object_name,
        material_name=material_name,
        geometry_delta=geometry_delta,
        base_color_texture_index=base_color_texture_index,
    )
    (root / "textured_mesh.obj").write_text(
        "\n".join(
            [
                "mtllib textured_mesh.mtl",
                f"v {0.0 + geometry_delta} 0.0 0.0",
                "v 1.0 0.0 0.0",
                "v 0.0 1.0 0.0",
                "vt 0.0 0.0",
                "vt 1.0 0.0",
                "vt 0.0 1.0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "textured_mesh.mtl").write_text(
        "\n".join(
            [
                "newmtl Material",
                "Kd 0.8 0.8 0.8",
                f"map_Kd {material_assignment}",
                "map_Pm textured_mesh_metallic.jpg",
                "map_Pr textured_mesh_roughness.jpg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_texture(root / "textured_mesh.jpg", albedo_pixel, metadata=metadata)
    write_texture(
        root / "textured_mesh_metallic.jpg", (40, 40, 40), metadata=metadata
    )
    write_texture(
        root / "textured_mesh_roughness.jpg", (180, 180, 180), metadata=metadata
    )
    return root


def test_different_glb_node_names_are_semantically_equal(tmp_path: Path) -> None:
    official = make_package(
        tmp_path / "official",
        object_name="textured_mesh",
        material_name="Material.001",
        metadata="official",
    )
    noop = make_package(
        tmp_path / "noop",
        object_name="textured_mesh.001",
        material_name="Material.002",
        metadata="noop",
    )

    comparison = compare_artifact_semantics(official, noop)

    assert comparison["semantically_equal"] is True
    assert comparison["raw_glb_sha256_equal"] is False


def test_changed_texture_pixels_are_rejected(tmp_path: Path) -> None:
    official = make_package(
        tmp_path / "official",
        object_name="mesh",
        material_name="material",
        metadata="a",
    )
    noop = make_package(
        tmp_path / "noop",
        object_name="mesh",
        material_name="material",
        metadata="b",
        albedo_pixel=(11, 20, 30),
    )

    assert compare_artifact_semantics(official, noop)["semantically_equal"] is False


def test_changed_geometry_is_rejected(tmp_path: Path) -> None:
    official = make_package(
        tmp_path / "official",
        object_name="mesh",
        material_name="material",
        metadata="a",
    )
    noop = make_package(
        tmp_path / "noop",
        object_name="mesh",
        material_name="material",
        metadata="b",
        geometry_delta=1e-4,
    )

    assert compare_artifact_semantics(official, noop)["semantically_equal"] is False


def test_changed_pbr_material_assignment_is_rejected(tmp_path: Path) -> None:
    official = make_package(
        tmp_path / "official",
        object_name="mesh",
        material_name="material",
        metadata="a",
    )
    noop = make_package(
        tmp_path / "noop",
        object_name="mesh",
        material_name="material",
        metadata="b",
        material_assignment="textured_mesh_roughness.jpg",
        base_color_texture_index=1,
    )

    assert compare_artifact_semantics(official, noop)["semantically_equal"] is False


def test_decoded_image_comparison_ignores_container_metadata(tmp_path: Path) -> None:
    official = make_package(
        tmp_path / "official",
        object_name="mesh",
        material_name="material",
        metadata="first metadata block",
    )
    noop = make_package(
        tmp_path / "noop",
        object_name="mesh",
        material_name="material",
        metadata="different metadata block",
    )
    assert (official / "textured_mesh.jpg").read_bytes() != (
        noop / "textured_mesh.jpg"
    ).read_bytes()

    assert compare_artifact_semantics(official, noop)["semantically_equal"] is True


def test_raw_glb_hashes_are_provenance_not_scientific_gate(tmp_path: Path) -> None:
    official = make_package(
        tmp_path / "official",
        object_name="textured_mesh",
        material_name="Material.001",
        metadata="a",
    )
    noop = make_package(
        tmp_path / "noop",
        object_name="textured_mesh.002",
        material_name="Material.003",
        metadata="b",
    )

    comparison = compare_artifact_semantics(official, noop)

    assert comparison["raw_glb_sha256_equal"] is False
    assert comparison["raw_glb_sha256_is_scientific_gate"] is False
    assert comparison["semantically_equal"] is True
