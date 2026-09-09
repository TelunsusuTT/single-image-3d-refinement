from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hy3dft.lora import targeting  # noqa: E402


class FakeWeight:
    shape = (4, 4)


class FakeLinear:
    def __init__(self) -> None:
        self.weight = FakeWeight()


class FakeModule:
    def __init__(self, **children: object) -> None:
        self._children = children

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, child in self._children.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if hasattr(child, "named_modules"):
                yield from child.named_modules(child_prefix)  # type: ignore[attr-defined]
            else:
                yield child_prefix, child


def projection() -> FakeModule:
    return FakeModule(
        to_q=FakeLinear(),
        to_k=FakeLinear(),
        to_v=FakeLinear(),
        to_out=FakeModule(**{"0": FakeLinear()}),
    )


def fake_hunyuan_unet() -> FakeModule:
    return FakeModule(
        block=FakeModule(
            attn_refview=projection(),
            attn_dino=projection(),
            attn_multiview=projection(),
            attn1=projection(),
            attn2=projection(),
            ff=FakeModule(**{"0": FakeLinear(), "1": FakeLinear()}),
            learned_text_clip=FakeLinear(),
        )
    )


@pytest.fixture(autouse=True)
def fake_linear_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        targeting,
        "_is_linear",
        lambda module: isinstance(module, FakeLinear),
    )


def test_reference_conditioning_preset_selects_only_intended_projections() -> None:
    targets = targeting.select_lora_targets(fake_hunyuan_unet(), preset="ref_dino")

    assert targets == [
        "block.attn_refview.to_q",
        "block.attn_refview.to_k",
        "block.attn_refview.to_v",
        "block.attn_refview.to_out.0",
        "block.attn_dino.to_q",
        "block.attn_dino.to_k",
        "block.attn_dino.to_v",
        "block.attn_dino.to_out.0",
    ]
    assert all("attn_multiview" not in name for name in targets)
    assert all("attn1" not in name for name in targets)
    assert all("attn2" not in name for name in targets)
    assert all("ff" not in name for name in targets)
    assert all("learned_text_clip" not in name for name in targets)


def test_target_validation_rejects_forbidden_or_non_projection_names() -> None:
    with pytest.raises(ValueError, match="attn_multiview"):
        targeting.validate_target_names(["block.attn_multiview.to_q"])

    with pytest.raises(ValueError, match="to_q"):
        targeting.validate_target_names(["block.attn_refview.norm"])


def test_empty_target_selection_fails_closed() -> None:
    with pytest.raises(ValueError, match="empty"):
        targeting.select_lora_targets(
            FakeModule(linear=FakeLinear()),
            preset="ref_dino",
        )
