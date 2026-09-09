"""Canonical semantic identity for textured-mesh artifact packages."""

from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any, Iterable


TEXTURE_FILES = {
    "albedo": "textured_mesh.jpg",
    "metallic": "textured_mesh_metallic.jpg",
    "roughness": "textured_mesh_roughness.jpg",
}
_COMPONENT_FORMAT = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class SemanticIdentityError(RuntimeError):
    """Raised when a required semantic artifact cannot be canonicalized."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SemanticIdentityError(f"Missing or empty {label}: {resolved}")
    return resolved


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    source = _require_file(path, "GLB")
    raw = source.read_bytes()
    if len(raw) < 20:
        raise SemanticIdentityError(f"GLB is truncated: {source}")
    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total_length != len(raw):
        raise SemanticIdentityError(f"Invalid GLB header: {source}")
    json_chunk: bytes | None = None
    binary_chunk: bytes | None = None
    offset = 12
    while offset < total_length:
        if offset + 8 > total_length:
            raise SemanticIdentityError(f"Invalid GLB chunk header: {source}")
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        end = offset + chunk_length
        if end > total_length:
            raise SemanticIdentityError(f"Invalid GLB chunk length: {source}")
        payload = raw[offset:end]
        offset = end
        if chunk_type == 0x4E4F534A:
            json_chunk = payload
        elif chunk_type == 0x004E4942:
            binary_chunk = payload
    if json_chunk is None or binary_chunk is None:
        raise SemanticIdentityError(f"GLB must contain JSON and BIN chunks: {source}")
    try:
        document = json.loads(json_chunk.decode("utf-8").rstrip(" \t\r\n\0"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticIdentityError(f"Invalid GLB JSON: {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise SemanticIdentityError(f"GLB JSON root must be an object: {source}")
    return document, binary_chunk


def _strip_exporter_labels(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_exporter_labels(item)
            for key, item in value.items()
            if key not in {"name", "generator"}
        }
    if isinstance(value, list):
        return [_strip_exporter_labels(item) for item in value]
    return value


def _accessor_values(
    document: dict[str, Any], binary: bytes, accessor_index: int
) -> list[Any]:
    try:
        accessor = document["accessors"][accessor_index]
        view = document["bufferViews"][accessor["bufferView"]]
        component_type = int(accessor["componentType"])
        component_format = _COMPONENT_FORMAT[component_type]
        component_count = _TYPE_COMPONENTS[str(accessor["type"])]
        count = int(accessor["count"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SemanticIdentityError(
            f"Invalid GLB accessor {accessor_index}: {exc}"
        ) from exc
    if "sparse" in accessor:
        raise SemanticIdentityError("Sparse GLB accessors are not supported")
    element_struct = struct.Struct("<" + component_format * component_count)
    stride = int(view.get("byteStride", element_struct.size))
    if stride < element_struct.size:
        raise SemanticIdentityError(f"Invalid accessor stride for {accessor_index}")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    values: list[Any] = []
    for row_index in range(count):
        offset = start + row_index * stride
        end = offset + element_struct.size
        if end > len(binary):
            raise SemanticIdentityError(f"Accessor {accessor_index} exceeds BIN chunk")
        row = element_struct.unpack_from(binary, offset)
        values.append(row[0] if component_count == 1 else list(row))
    return values


def _decoded_image(payload: bytes, label: str) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            mode = image.mode
            dimensions = list(image.size)
            pixels = image.tobytes()
    except Exception as exc:
        raise SemanticIdentityError(f"Could not decode {label}: {exc}") from exc
    return {
        "mode": mode,
        "dimensions": dimensions,
        "pixel_array_sha256": sha256_bytes(pixels),
    }


def _glb_images(document: dict[str, Any], binary: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    views = document.get("bufferViews", [])
    for index, image in enumerate(document.get("images", [])):
        if not isinstance(image, dict) or "bufferView" not in image:
            raise SemanticIdentityError(
                f"GLB image {index} must use an embedded bufferView"
            )
        try:
            view = views[int(image["bufferView"])]
            start = int(view.get("byteOffset", 0))
            end = start + int(view["byteLength"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SemanticIdentityError(f"Invalid GLB image {index}: {exc}") from exc
        rows.append(
            {
                "mime_type": image.get("mimeType"),
                **_decoded_image(binary[start:end], f"GLB image {index}"),
            }
        )
    return rows


def glb_semantic_payload(path: Path) -> dict[str, Any]:
    """Return GLB semantics while ignoring exporter labels and container layout."""

    document, binary = _read_glb(path)
    primitive_rows: list[dict[str, Any]] = []
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = {
                semantic: {
                    "count": len(values := _accessor_values(document, binary, int(index))),
                    "values_sha256": canonical_sha256(values),
                }
                for semantic, index in sorted(primitive.get("attributes", {}).items())
            }
            indices = (
                _accessor_values(document, binary, int(primitive["indices"]))
                if "indices" in primitive
                else []
            )
            primitive_rows.append(
                {
                    "attributes": attributes,
                    "indices_count": len(indices),
                    "indices_sha256": canonical_sha256(indices),
                    "material": primitive.get("material"),
                    "mode": int(primitive.get("mode", 4)),
                }
            )
    primitive_rows.sort(key=canonical_sha256)
    return {
        "mesh_count": len(document.get("meshes", [])),
        "primitive_count": len(primitive_rows),
        "primitives": primitive_rows,
        "scene_semantics": _strip_exporter_labels(
            {
                "scene": document.get("scene"),
                "scenes": document.get("scenes", []),
                "nodes": document.get("nodes", []),
            }
        ),
        "material_count": len(document.get("materials", [])),
        "materials": _strip_exporter_labels(document.get("materials", [])),
        "textures": _strip_exporter_labels(document.get("textures", [])),
        "samplers": _strip_exporter_labels(document.get("samplers", [])),
        "embedded_images": _glb_images(document, binary),
    }


def _hash_rows(rows: Iterable[Any]) -> str:
    return canonical_sha256(list(rows))


def obj_semantic_payload(path: Path) -> dict[str, Any]:
    source = _require_file(path, "OBJ")
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    normals: list[list[float]] = []
    faces: list[list[str]] = []
    material_assignments: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] == "v":
            vertices.append([float(value) for value in parts[1:]])
        elif parts[0] == "vt":
            uvs.append([float(value) for value in parts[1:]])
        elif parts[0] == "vn":
            normals.append([float(value) for value in parts[1:]])
        elif parts[0] == "f":
            faces.append(parts[1:])
        elif parts[0] == "usemtl":
            material_assignments.append(" ".join(parts[1:]))
    return {
        "vertex_count": len(vertices),
        "vertices_sha256": _hash_rows(vertices),
        "uv_count": len(uvs),
        "uvs_sha256": _hash_rows(uvs),
        "normal_count": len(normals),
        "normals_sha256": _hash_rows(normals),
        "face_count": len(faces),
        "faces_sha256": _hash_rows(faces),
        "material_assignments": material_assignments,
    }


def mtl_semantic_payload(path: Path) -> dict[str, Any]:
    source = _require_file(path, "MTL")
    rows: list[list[str]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append(stripped.split())
    return {
        "material_count": sum(row[0] == "newmtl" for row in rows),
        "scalar_and_texture_assignments": rows,
    }


def decoded_file_image(path: Path) -> dict[str, Any]:
    source = _require_file(path, "texture image")
    return _decoded_image(source.read_bytes(), source.name)


def artifact_semantic_fingerprint(artifact_dir: Path) -> dict[str, Any]:
    """Fingerprint geometry, UVs, PBR assignments, and decoded texture pixels."""

    root = artifact_dir.expanduser().resolve()
    glb_path = _require_file(root / "textured_mesh.glb", "textured GLB")
    scientific_payload = {
        "glb": glb_semantic_payload(glb_path),
        "obj": obj_semantic_payload(root / "textured_mesh.obj"),
        "mtl": mtl_semantic_payload(root / "textured_mesh.mtl"),
        "decoded_textures": {
            role: decoded_file_image(root / filename)
            for role, filename in TEXTURE_FILES.items()
        },
    }
    return {
        "artifact_dir": str(root),
        "raw_glb_sha256": sha256_file(glb_path),
        "raw_glb_sha256_is_provenance_only": True,
        "scientific_payload": scientific_payload,
        "semantic_sha256": canonical_sha256(scientific_payload),
    }


def compare_artifact_semantics(
    baseline_dir: Path, no_op_dir: Path
) -> dict[str, Any]:
    baseline = artifact_semantic_fingerprint(baseline_dir)
    no_op = artifact_semantic_fingerprint(no_op_dir)
    return {
        "semantically_equal": (
            baseline["semantic_sha256"] == no_op["semantic_sha256"]
            and baseline["scientific_payload"] == no_op["scientific_payload"]
        ),
        "raw_glb_sha256_equal": (
            baseline["raw_glb_sha256"] == no_op["raw_glb_sha256"]
        ),
        "raw_glb_sha256_is_scientific_gate": False,
        "baseline": baseline,
        "gating_no_op": no_op,
    }
