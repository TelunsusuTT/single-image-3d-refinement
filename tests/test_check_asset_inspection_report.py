from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_asset_inspection_report import main  # noqa: E402


def valid_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "input_glb": "fake.glb",
        "import_status": "OK",
        "object_count": 1,
        "mesh_object_count": 1,
        "material_count": 1,
        "texture_image_count": 1,
        "total_vertices": 8,
        "total_polygons": 6,
        "total_triangles_estimate": 12,
        "bounding_box_min": [0.0, 0.0, 0.0],
        "bounding_box_max": [1.0, 1.0, 1.0],
        "per_object": [
            {
                "name": "mesh",
                "type": "MESH",
                "vertex_count": 8,
                "polygon_count": 6,
                "uv_layer_count": 1,
                "material_slot_count": 1,
            }
        ],
        "per_material": [
            {
                "name": "material",
                "use_nodes": True,
                "image_texture_names": ["texture.png"],
            }
        ],
        "warnings": {
            "no_mesh": False,
            "no_uv": False,
            "no_material": False,
            "no_texture_image": False,
        },
    }
    report.update(overrides)
    return report


def write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def run_report(report: dict[str, object]) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        report_json = Path(tmpdir) / "asset_inspection.json"
        write_report(report_json, report)
        return main(["--report-json", str(report_json)])


def test_valid_report_passes() -> None:
    assert run_report(valid_report()) == 0


def test_no_mesh_fails() -> None:
    report = valid_report(mesh_object_count=0, total_vertices=0, total_polygons=0)
    assert run_report(report) == 1


def test_no_uv_fails() -> None:
    report = valid_report(
        per_object=[
            {
                "name": "mesh",
                "type": "MESH",
                "vertex_count": 8,
                "polygon_count": 6,
                "uv_layer_count": 0,
                "material_slot_count": 1,
            }
        ],
    )
    assert run_report(report) == 1


def test_no_material_fails() -> None:
    report = valid_report(material_count=0, per_material=[])
    assert run_report(report) == 1


def test_no_texture_warns_but_passes() -> None:
    report = valid_report(texture_image_count=0)
    assert run_report(report) == 0
