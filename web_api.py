# web_api.py
import asyncio
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import onnxruntime

from modules.processors.frame import face_swapper
from modules.face_analyser import get_one_face
from modules.core import decode_execution_providers, suggest_default_execution_provider
import modules.globals

app = FastAPI(title="Deep-Live-Cam Web API")

current_source_face = None
_models_loaded = False


def init_globals():
    """Initialize global settings and load models."""
    global _models_loaded
    if _models_loaded:
        return

    if not modules.globals.execution_providers:
        best_provider = suggest_default_execution_provider()
        modules.globals.execution_providers = decode_execution_providers([best_provider])
    
    print(f"[INFO] Using execution providers: {modules.globals.execution_providers}")
    
    modules.globals.frame_processors = ['face_swapper']
    modules.globals.many_faces = False
    modules.globals.opacity = 1.0
    modules.globals.mouth_mask = False
    modules.globals.poisson_blend = False
    modules.globals.sharpness = 0.0
    modules.globals.fp_ui = {'face_enhancer': False, 'face_enhancer_gpen256': False, 'face_enhancer_gpen512': False}

    face_swapper.pre_check()
    face_swapper.pre_start()

    _models_loaded = True


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    await asyncio.to_thread(init_globals)
    print("Deep-Live-Cam models loaded and ready")


@app.get("/")
async def get_index():
    """返回主页面"""
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/set_source_face")
async def set_source_face(image_data: dict):
    """设置源脸图片（Base64 格式）"""
    global current_source_face

    img_data = base64.b64decode(image_data["image"].split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    current_source_face = await asyncio.to_thread(get_one_face, source_image)

    if current_source_face is None:
        return {"status": "error", "message": "No face detected in source image"}
    return {"status": "success", "message": "Source face set"}


@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    """WebSocket 处理实时视频流"""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()

            img_data = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if current_source_face is not None:
                try:
                    result_frame = await asyncio.to_thread(
                        face_swapper.process_frame,
                        current_source_face,
                        frame
                    )
                except Exception as e:
                    result_frame = frame
                    print(f"Swap face error: {e}")
            else:
                result_frame = frame

            _, buffer = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            result_base64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(f"data:image/jpeg;base64,{result_base64}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
