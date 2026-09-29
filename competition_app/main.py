from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .depth_pro_backend import DepthProUnavailableError
from .inference import BokehPipeline


STATIC_DIR = Path(__file__).resolve().parent / "static"
PROJECT_DIR = Path(__file__).resolve().parents[1]
PAPER_PATH = PROJECT_DIR / "docs" / "main_arxiv.pdf"
MAX_UPLOAD_BYTES = int(os.getenv("BOKEH_MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
SHOWCASE_FILES = {
    "street-original": PROJECT_DIR / "94.png",
    "street-result": STATIC_DIR / "showcase" / "street-physical.png",
}
SOURCE_FILES = {
    "web-service": PROJECT_DIR / "competition_app" / "main.py",
    "inference-pipeline": PROJECT_DIR / "competition_app" / "inference.py",
    "depthpro-adapter": PROJECT_DIR / "competition_app" / "depth_pro_backend.py",
    "explicit-renderer": PROJECT_DIR / "method" / "explicit_renderer.py",
    "residual-corrector": PROJECT_DIR / "method" / "corrector.py",
    "repair-prior": PROJECT_DIR / "method" / "repair_prior.py",
}


class ApiError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


async def read_image_upload(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_TYPES:
        raise ApiError("请选择 JPG、PNG、WebP、BMP 或 TIFF 图片。", "UNSUPPORTED_IMAGE")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ApiError("上传的图片为空。", "EMPTY_IMAGE")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ApiError(
            f"图片不能超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB。",
            "IMAGE_TOO_LARGE",
            413,
        )
    return data


def pipeline_api_error(exc: Exception, pipeline: BokehPipeline) -> ApiError:
    if isinstance(exc, DepthProUnavailableError):
        return ApiError(str(exc), "DEPTH_PRO_UNAVAILABLE", 503)
    if isinstance(exc, FileNotFoundError):
        return ApiError(str(exc), "MODEL_NOT_FOUND", 503)
    if isinstance(exc, ValueError):
        code = "DEPTH_RESULT_NOT_FOUND" if "深度结果已失效" in str(exc) else "INVALID_INPUT"
        return ApiError(str(exc), code, 409 if code == "DEPTH_RESULT_NOT_FOUND" else 400)
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        pipeline.recover_from_oom()
        return ApiError(
            "显存不足。请在工作台选择更小的处理尺寸后重试；必要时关闭其他占用 GPU 的程序。",
            "OUT_OF_MEMORY",
            507,
        )
    return ApiError(f"推理失败：{exc}", "INFERENCE_FAILED", 500)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.pipeline = BokehPipeline()
    yield


app = FastAPI(
    title="无界光圈",
    description="面向后期景深控制的计算摄影平台",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
# Make imports and lightweight ASGI tests useful even when the lifespan hook is
# not driven by the test client. Models remain lazy and are not loaded here.
app.state.pipeline = BokehPipeline()


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "服务暂时无法完成请求，请检查控制台日志后重试。",
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "请求参数不完整或格式不正确。",
                "details": json.loads(exc.json()),
            }
        },
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/studio", include_in_schema=False)
async def studio_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "studio.html")


@app.get("/research", include_in_schema=False)
async def research_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "research.html")


@app.get("/open-source", include_in_schema=False)
async def open_source_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "open-source.html")


@app.get("/deploy", include_in_schema=False)
async def deploy_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "deploy.html")


@app.get("/paper", include_in_schema=False)
async def paper_file() -> FileResponse:
    if not PAPER_PATH.is_file():
        raise ApiError("论文 PDF 尚未放入 docs 目录。", "PAPER_NOT_FOUND", 404)
    return FileResponse(
        PAPER_PATH,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="main_arxiv.pdf"'},
    )


@app.get("/api/showcase/{asset_name}", include_in_schema=False)
async def showcase_asset(asset_name: str) -> FileResponse:
    path = SHOWCASE_FILES.get(asset_name)
    if path is None or not path.is_file():
        raise ApiError("案例图片不存在。", "SHOWCASE_NOT_FOUND", 404)
    return FileResponse(
        path,
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@app.get("/api/source/{source_name}", include_in_schema=False)
async def source_file(source_name: str) -> FileResponse:
    path = SOURCE_FILES.get(source_name)
    if path is None or not path.is_file():
        raise ApiError("源码文件不存在。", "SOURCE_NOT_FOUND", 404)
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@app.get("/api/health")
async def health(request: Request) -> dict:
    pipeline: BokehPipeline = request.app.state.pipeline
    return {"status": "ok", "service": "WujieAperture", **pipeline.readiness()}


@app.post(
    "/api/depth",
    responses={200: {"content": {"image/png": {}}}},
)
async def estimate_depth(
    request: Request,
    image: UploadFile = File(...),
    max_long_side: int | None = Form(None),
) -> Response:
    data = await read_image_upload(image)
    pipeline: BokehPipeline = request.app.state.pipeline
    try:
        decoded = await run_in_threadpool(pipeline.decode_image, data)
        result = await run_in_threadpool(
            pipeline.estimate_depth,
            decoded,
            max_long_side,
        )
    except (DepthProUnavailableError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise pipeline_api_error(exc, pipeline) from exc

    stats = result.depth_stats
    headers = {
        "Cache-Control": "no-store",
        "X-Wujie-Aperture-Depth-Id": result.depth_id,
        "X-Wujie-Aperture-Width": str(result.width),
        "X-Wujie-Aperture-Height": str(result.height),
        "X-Wujie-Aperture-Depth-Time": f"{result.elapsed_seconds:.3f}",
        "X-Wujie-Aperture-Depth-Min": f"{stats.get('metric_depth_min_m', 0.0):.4f}",
        "X-Wujie-Aperture-Depth-Max": f"{stats.get('metric_depth_max_m', 0.0):.4f}",
    }
    return Response(result.image_bytes, media_type="image/png", headers=headers)


@app.post(
    "/api/render",
    responses={200: {"content": {"image/png": {}}}},
)
async def render(
    request: Request,
    image: UploadFile = File(...),
    f_number: float = Form(1.8),
    engine: str = Form("vatd"),
    focus_x: float | None = Form(None),
    focus_y: float | None = Form(None),
    depth_id: str | None = Form(None),
    max_long_side: int | None = Form(None),
) -> Response:
    data = await read_image_upload(image)
    if focus_x is None and focus_y is None:
        raise ApiError("请先点击图片指定目标焦平面。", "FOCUS_REQUIRED")
    if (focus_x is None) != (focus_y is None):
        raise ApiError("焦点坐标必须同时包含横纵坐标。", "INVALID_FOCUS")

    pipeline: BokehPipeline = request.app.state.pipeline
    try:
        decoded = await run_in_threadpool(pipeline.decode_image, data)
        focus = (focus_x, focus_y)
        result = await run_in_threadpool(
            pipeline.render,
            decoded,
            f_number,
            focus,
            engine,
            depth_id=depth_id,
            max_long_side=max_long_side,
        )
    except (DepthProUnavailableError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise pipeline_api_error(exc, pipeline) from exc

    headers = {
        "Cache-Control": "no-store",
        "X-Wujie-Aperture-Width": str(result.width),
        "X-Wujie-Aperture-Height": str(result.height),
        "X-Wujie-Aperture-F-Number": f"{result.f_number:g}",
        "X-Wujie-Aperture-Engine": result.engine.upper(),
        "X-Wujie-Aperture-Output": result.output_mode,
        "X-Wujie-Aperture-Focus": result.focus_mode,
        "X-Wujie-Aperture-Elapsed": f"{result.elapsed_seconds:.3f}",
        "X-Wujie-Aperture-Depth-Time": f"{result.depth_seconds:.3f}",
        "X-Wujie-Aperture-Render-Time": f"{result.render_seconds:.3f}",
    }
    return Response(result.image_bytes, media_type="image/png", headers=headers)
