# web_api.py
import asyncio
import base64
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
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
_thread_pool = ThreadPoolExecutor(max_workers=2)

FACE_DETECTION_CACHE = {}
CACHE_FRAMES = 3

_web_args = None


def set_web_args(args):
    global _web_args
    _web_args = args


def init_globals():
    global _models_loaded
    if _models_loaded:
        return

    if _web_args and _web_args.execution_provider:
        modules.globals.execution_providers = decode_execution_providers(_web_args.execution_provider)
    elif not modules.globals.execution_providers:
        best_provider = suggest_default_execution_provider()
        modules.globals.execution_providers = decode_execution_providers([best_provider])
    
    print(f"[INFO] Using execution providers: {modules.globals.execution_providers}")
    
    modules.globals.frame_processors = ['face_swapper']
    
    if _web_args:
        modules.globals.many_faces = _web_args.many_faces
        modules.globals.opacity = _web_args.opacity
        modules.globals.mouth_mask = _web_args.mouth_mask
        modules.globals.poisson_blend = _web_args.poisson_blend
        modules.globals.sharpness = _web_args.sharpness
    else:
        modules.globals.many_faces = False
        modules.globals.opacity = 1.0
        modules.globals.mouth_mask = False
        modules.globals.poisson_blend = False
        modules.globals.sharpness = 0.0
    
    modules.globals.fp_ui = {'face_enhancer': False, 'face_enhancer_gpen256': False, 'face_enhancer_gpen512': False}

    if _web_args:
        print(f"[INFO] Web args: mouth_mask={_web_args.mouth_mask}, many_faces={_web_args.many_faces}, opacity={_web_args.opacity}")

    face_swapper.pre_check()
    face_swapper.pre_start()

    _models_loaded = True


@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(init_globals)
    print("Deep-Live-Cam models loaded and ready")


@app.get("/")
async def get_index():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/set_source_face")
async def set_source_face(image_data: dict):
    global current_source_face

    img_data = base64.b64decode(image_data["image"].split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    current_source_face = await asyncio.to_thread(get_one_face, source_image)

    if current_source_face is None:
        return {"status": "error", "message": "No face detected in source image"}
    return {"status": "success", "message": "Source face set"}


def process_frame_with_cache(source_face, frame, cached_face, frame_count):
    if cached_face is not None and frame_count % CACHE_FRAMES != 0:
        try:
            result = face_swapper.process_frame(source_face, frame, target_face=cached_face)
            return result, cached_face
        except Exception:
            pass

    target_face = get_one_face(frame)
    if target_face is None:
        return frame, cached_face

    result = face_swapper.process_frame(source_face, frame, target_face=target_face)
    return result, target_face


@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    await websocket.accept()

    cached_target_face = None
    frame_count = 0
    last_time = time.time()

    try:
        while True:
            data = await websocket.receive_bytes()

            t0 = time.time()

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            t1 = time.time()

            if current_source_face is not None:
                try:
                    result_frame, cached_target_face = await asyncio.get_event_loop().run_in_executor(
                        _thread_pool,
                        process_frame_with_cache,
                        current_source_face,
                        frame,
                        cached_target_face,
                        frame_count
                    )
                except Exception as e:
                    result_frame = frame
                    cached_target_face = None
                    print(f"Swap face error: {e}")
            else:
                result_frame = frame

            t2 = time.time()

            _, buffer = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

            t3 = time.time()

            await websocket.send_bytes(buffer.tobytes())

            t4 = time.time()

            frame_count += 1
            if frame_count % 10 == 0:
                fps = 1.0 / (t4 - last_time)
                last_time = t4
                print(f"FPS: {fps:.1f} | Decode: {(t1-t0)*1000:.0f}ms | Swap: {(t2-t1)*1000:.0f}ms | Encode: {(t3-t2)*1000:.0f}ms | Send: {(t4-t3)*1000:.0f}ms")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
