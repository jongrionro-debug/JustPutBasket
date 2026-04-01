from switch_query import tagging, v1
from switch_query.image_module import InventoryRow, V1Pipeline, V1PipelineConfig
from switch_query.image_module.local_vlm_tagger import _coerce_json
from switch_query.image_module.preprocessing import build_image_inventory


def test_legacy_image_module_reexports_new_packages() -> None:
    assert build_image_inventory is tagging.build_image_inventory
    assert InventoryRow is tagging.InventoryRow
    assert V1Pipeline is v1.V1Pipeline
    assert V1PipelineConfig is v1.V1PipelineConfig


def test_legacy_local_vlm_wrapper_exposes_json_coercer() -> None:
    payload = _coerce_json("not json")

    assert payload["review_needed"] is True
