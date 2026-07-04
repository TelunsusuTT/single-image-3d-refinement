from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_phase2k4_render_eval_configs import main as config_main  # noqa: E402


ASSET_IDS = ["B075YLTF7Q", "B07HSK7MXZ", "B073NZS57V", "B07B8MWCR8"]


def write_cases_config(path: Path, root: Path) -> Path:
    data = {
        "output_root": str(root / "outputs" / "phase2k4"),
        "asset_ids": ASSET_IDS,
        "input_views": ["004", "005"],
        "assets": {asset_id: {} for asset_id in ASSET_IDS},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_render_eval_configs_are_generated() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cases_config = write_cases_config(root / "cases.json", root)
        assert config_main(["--cases-config", str(cases_config)]) == 0

        output_root = root / "outputs" / "phase2k4"
        for input_view in ("004", "005"):
            path = output_root / "render_eval_configs" / f"render_eval_input_{input_view}.json"
            assert path.is_file()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert list(data["cases"].keys()) == ASSET_IDS
            assert data["view_ids"] == ["000", "001", "002", "003", "004", "005"]
            assert data["output_root"].endswith(f"rendered_eval/input_{input_view}")
            for asset_id, case in data["cases"].items():
                assert f"reference_view_ablation_truepbr200" not in case["base_glb"]
                assert f"input_{input_view}" in case["base_glb"]
                assert f"input_{input_view}" in case["finetuned_glb"]
                assert case["base_glb"].endswith("base_textured_mesh.glb")
                assert case["finetuned_glb"].endswith("finetuned_textured_mesh.glb")
