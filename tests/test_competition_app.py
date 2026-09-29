from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import render_photos
from competition_app.inference import (
    BokehPipeline,
    DepthEstimateResult,
    PipelineSettings,
    RenderResult,
)
from competition_app.main import PAPER_PATH, SHOWCASE_FILES, app


def make_png(width: int = 96, height: int = 72) -> bytes:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(10, 240, width, dtype=np.uint8)
    array[..., 1] = 100
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_accepts_valid_png() -> None:
    image = BokehPipeline.decode_image(make_png())
    assert image.mode == "RGB"
    assert image.size == (96, 72)


def test_decode_image_rejects_small_image() -> None:
    with pytest.raises(ValueError, match="尺寸过小"):
        BokehPipeline.decode_image(make_png(32, 32))


def test_index_and_health_are_available() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "无界光圈" in response.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["service"] == "WujieAperture"
    assert health.json()["models"]["vatd"]["available"] is True
    assert health.json()["models"]["ebb"]["available"] is True
    assert health.json()["depth_pro_package"] is True
    assert health.json()["depth_pro_source"] in {"local", "installed"}

    for route, title in [
        ("/studio", "无界光圈工作台"),
        ("/research", "散景可控，方法可查"),
        ("/open-source", "开放路线，发布准备中"),
        ("/deploy", "让散景渲染在本地运行"),
    ]:
        page = client.get(route)
        assert page.status_code == 200
        assert title in page.text
        assert "/assets/aperture-mark.svg" in page.text
        assert "/assets/wujie-aperture.css" in page.text
        assert "焦域" not in page.text
        assert "FocusField" not in page.text
        assert "DepthLens" not in page.text

    favicon = client.get("/assets/aperture-mark.svg")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"].startswith("image/svg+xml")

    studio = client.get("/studio")
    assert "智能主体" not in studio.text
    assert "点击画面指定目标焦平面并开始物理渲染" in studio.text
    assert 'role="slider"' in studio.text
    assert 'id="customSizeInput"' in studio.text
    assert '<option value="1920">' in studio.text
    assert '<option value="custom">' in studio.text

    paper = client.get("/paper")
    if PAPER_PATH.is_file():
        assert paper.status_code == 200
        assert paper.headers["content-type"] == "application/pdf"
        assert paper.content.startswith(b"%PDF")
    else:
        assert paper.status_code == 404
        assert paper.json()["error"]["code"] == "PAPER_NOT_FOUND"

    source = client.get("/api/source/inference-pipeline")
    assert source.status_code == 200
    assert "class BokehPipeline" in source.text


def test_ebb_engine_rejects_unverified_aperture() -> None:
    with pytest.raises(ValueError, match="EBB 专项引擎仅在 f/1.8"):
        BokehPipeline._validate_parameters(2.8, None, "ebb")


def test_processing_size_override_preserves_aspect_ratio_and_is_validated() -> None:
    settings = PipelineSettings(
        device_name="cpu",
        max_long_side=320,
        max_custom_long_side=512,
    )
    pipeline = BokehPipeline(settings)
    image = Image.new("RGB", (640, 320), color=(10, 20, 30))

    resized = pipeline._prepare_image(image, 320)
    assert resized.size == (320, 160)

    with pytest.raises(ValueError, match="不能小于 256"):
        pipeline._prepare_image(image, 252)
    with pytest.raises(ValueError, match="不能超过 512"):
        pipeline._prepare_image(image, 516)
    with pytest.raises(ValueError, match="4 的倍数"):
        pipeline._prepare_image(image, 257)


def test_depth_endpoint_rejects_unsafe_custom_processing_size() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/depth",
        files={"image": ("sample.png", make_png(), "image/png")},
        data={"max_long_side": "4096"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_INPUT"
    assert "不能超过" in response.json()["error"]["message"]


def test_render_endpoint_returns_only_physical_png(monkeypatch: pytest.MonkeyPatch) -> None:
    output = make_png(96, 72)
    captured: dict[str, object] = {}

    def fake_render(*args, **kwargs) -> RenderResult:
        captured["max_long_side"] = kwargs.get("max_long_side")
        return RenderResult(
            image_bytes=output,
            width=96,
            height=72,
            f_number=1.8,
            focus_mode="auto",
            elapsed_seconds=0.2,
            depth_seconds=0.1,
            render_seconds=0.1,
            depth_stats={},
            engine="ebb",
        )

    monkeypatch.setattr(app.state.pipeline, "render", fake_render)
    client = TestClient(app)
    response = client.post(
        "/api/render",
        files={"image": ("sample.png", make_png(), "image/png")},
        data={
            "f_number": "1.8",
            "engine": "ebb",
            "focus_x": "0.5",
            "focus_y": "0.5",
            "max_long_side": "1024",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-wujie-aperture-f-number"] == "1.8"
    assert response.headers["x-wujie-aperture-engine"] == "EBB"
    assert response.headers["x-wujie-aperture-output"] == "physical"
    assert captured["max_long_side"] == 1024
    assert response.content == output


def test_home_showcase_uses_physical_asset() -> None:
    physical_showcase = SHOWCASE_FILES["street-result"]
    assert physical_showcase.name == "street-physical.png"
    assert "final" not in physical_showcase.name.lower()
    assert physical_showcase.is_file()

    response = TestClient(app).get("/api/showcase/street-result")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_pipeline_uses_coarse_even_if_an_output_field_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PhysicalRenderer(torch.nn.Module):
        def forward(self, source, depth, f_number, focus_depth=None):
            return {
                "coarse": source * 0.5,
                "output": source * 0.9,
            }

    settings = PipelineSettings(device_name="cpu", max_long_side=96)
    pipeline = BokehPipeline(settings)
    depth = np.ones((72, 96), dtype=np.float32)
    monkeypatch.setattr(
        pipeline._depth,
        "predict_inverse_depth",
        lambda image: (depth, {}),
    )
    monkeypatch.setattr(
        pipeline,
        "load",
        lambda engine="vatd": pipeline._renderers.setdefault(engine, PhysicalRenderer()),
    )

    result = pipeline.render(
        BokehPipeline.decode_image(make_png()),
        f_number=1.8,
        focus_point=(0.5, 0.5),
    )

    with Image.open(io.BytesIO(result.image_bytes)) as rendered:
        assert rendered.size == (96, 72)
        assert np.asarray(rendered)[0, 0, 1] in {49, 50}


def test_pipeline_never_falls_back_to_final_output() -> None:
    final_only = {"output": torch.ones(1, 3, 8, 8)}
    with pytest.raises(RuntimeError, match="避免误用神经最终结果"):
        BokehPipeline._physical_tensor(final_only)


def test_physical_loader_does_not_construct_full_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    checkpoint = {
        "model": {
            "renderer.weight": torch.full_like(renderer.weight, 0.25),
            "corrector.weight": torch.ones(1),
        },
        "cfg": {"renderer_mode": "layered"},
    }
    monkeypatch.setattr(render_photos.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(render_photos, "build_renderer", lambda cfg, device: renderer)

    def fail_if_full_model_is_built(*args, **kwargs):
        raise AssertionError("网站物理链路不应构造完整神经模型")

    monkeypatch.setattr(render_photos, "build_model", fail_if_full_model_is_built)
    loaded, _ = render_photos.load_physical_renderer(
        checkpoint_path=render_photos.Path("unused.pth"),
        config_path="",
        weights="raw",
        device=torch.device("cpu"),
        overrides=[],
    )

    assert loaded is renderer
    assert torch.allclose(renderer.weight, torch.full_like(renderer.weight, 0.25))


def test_depth_endpoint_returns_preview_and_reusable_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = make_png(96, 72)

    def fake_estimate_depth(*args, **kwargs) -> DepthEstimateResult:
        return DepthEstimateResult(
            image_bytes=preview,
            depth_id="depth-test-id",
            width=96,
            height=72,
            elapsed_seconds=0.12,
            depth_stats={
                "metric_depth_min_m": 0.8,
                "metric_depth_max_m": 12.5,
            },
        )

    monkeypatch.setattr(app.state.pipeline, "estimate_depth", fake_estimate_depth)
    client = TestClient(app)
    response = client.post(
        "/api/depth",
        files={"image": ("sample.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-wujie-aperture-depth-id"] == "depth-test-id"
    assert response.headers["x-wujie-aperture-depth-min"] == "0.8000"
    assert response.content == preview


def test_depth_preview_is_cached_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = PipelineSettings(
        device_name="cpu",
        max_long_side=96,
        depth_cache_size=1,
    )
    pipeline = BokehPipeline(settings)
    depth = np.linspace(0.0, 1.0, 96 * 72, dtype=np.float32).reshape(72, 96)
    monkeypatch.setattr(
        pipeline._depth,
        "predict_inverse_depth",
        lambda image: (depth, {"metric_depth_min_m": 1.0, "metric_depth_max_m": 9.0}),
    )

    first = pipeline.estimate_depth(BokehPipeline.decode_image(make_png()))
    with Image.open(io.BytesIO(first.image_bytes)) as preview:
        assert preview.mode == "L"
        assert preview.size == (96, 72)
    assert pipeline._cached_depth(first.depth_id).normalized_depth is depth
    with pytest.raises(ValueError, match="深度结果与当前图片不匹配"):
        pipeline.render(
            Image.new("RGB", (96, 72), color=(240, 20, 20)),
            f_number=1.8,
            depth_id=first.depth_id,
        )

    second = pipeline.estimate_depth(BokehPipeline.decode_image(make_png()))
    assert second.depth_id != first.depth_id
    with pytest.raises(ValueError, match="深度结果已失效"):
        pipeline._cached_depth(first.depth_id)


def test_render_endpoint_rejects_wrong_file_type() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/render",
        files={"image": ("sample.txt", b"hello", "text/plain")},
        data={"f_number": "1.8"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSUPPORTED_IMAGE"


def test_render_endpoint_requires_clicked_focus() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/render",
        files={"image": ("sample.png", make_png(), "image/png")},
        data={"f_number": "1.8", "engine": "vatd"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "FOCUS_REQUIRED"
