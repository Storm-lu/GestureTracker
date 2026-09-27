"""MediaPipe 新版(Tasks API) 手部关键点检测 —— HandLandmarker。

新版不再使用 mediapipe.solutions.hands.Hands，而是使用
mediapipe.tasks.python.vision.HandLandmarker，需要一个 .task 模型文件。

依赖：
    pip install mediapipe opencv-python
"""

import os

# 屏蔽 MediaPipe / glog 的内部 INFO、WARNING 日志（需在 import mediapipe 之前设置）
os.environ.setdefault("GLOG_minloglevel", "3")

import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_module

# ------------------------------ 配置区 ------------------------------
# 待检测的静态图片列表（留空则跳过图片模式）
IMAGE_FILES = []

# 模型文件（放在脚本同目录下，不存在时自动下载）
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# 结果图片输出目录
OUTPUT_DIR = os.path.join(MODEL_DIR, "output")

NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# 手关键点连线（新版从 HandLandmarksConnections 获取）
HAND_CONNECTIONS = hand_landmarker_module.HandLandmarksConnections.HAND_CONNECTIONS

# 21 个关键点中：食指指尖的索引
INDEX_FINGER_TIP = hand_landmarker_module.HandLandmark.INDEX_FINGER_TIP

# 测距参数（针孔相机模型：D = f * W / w_px）
CAMERA_FOV_DEG = 60.0    # 摄像头水平视场角，按实际摄像头修改（常见 60°~70°）
FOCAL_LENGTH_PX = None   # 若已知焦距(像素)可直接填写；None 表示由视场角推算
PALM_WIDTH_M = 0.08      # 手掌(手腕 0 号 -> 食指根 5 号)实际宽度，单位米
DIST_SMOOTHING = 0.5     # 距离显示平滑系数，0~1，越小越平滑（1 表示不平滑）
# -------------------------------------------------------------------


def ensure_model(model_path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    """确保模型文件存在，不存在则从官方地址下载。"""
    if not os.path.exists(model_path):
        print(f"未找到模型文件，正在下载: {url}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        print(f"模型已下载到: {model_path}")
    return model_path


def create_landmarker(running_mode, num_hands: int = NUM_HANDS) -> vision.HandLandmarker:
    """按指定运行模式创建 HandLandmarker。"""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks(image, hand_landmarks) -> None:
    """在 BGR 图像上绘制关键点与骨骼连线。"""
    image_height, image_width, _ = image.shape
    points = [
        (int(landmark.x * image_width), int(landmark.y * image_height))
        for landmark in hand_landmarks
    ]
    for connection in HAND_CONNECTIONS:
        cv2.line(
            image,
            points[connection.start],
            points[connection.end],
            (0, 255, 0),
            2,
        )
    for point in points:
        cv2.circle(image, point, 4, (255, 0, 0), -1)


def focal_length_px(frame_width: int) -> float:
    """由水平视场角推算焦距（像素）。"""
    if FOCAL_LENGTH_PX:
        return float(FOCAL_LENGTH_PX)
    return (frame_width / 2.0) / math.tan(math.radians(CAMERA_FOV_DEG) / 2.0)


def estimate_distance_m(hand_landmarks, image_width: int, image_height: int,
                        focal_px: float):
    """针孔模型估算手到相机的距离(米)：D = f * W / w_px。

    以手腕(0 号)到食指根(5 号)的像素宽度作为尺子。
    """
    wrist = hand_landmarks[0]
    index_mcp = hand_landmarks[5]
    palm_px = math.hypot(
        (wrist.x - index_mcp.x) * image_width,
        (wrist.y - index_mcp.y) * image_height,
    )
    if palm_px < 1e-6:
        return None
    return focal_px * PALM_WIDTH_M / palm_px


def draw_info_panel(image, lines) -> None:
    """在图像左上角绘制半透明信息面板（仅支持 ASCII，cv2 无法渲染中文）。"""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad, line_height = 0.55, 1, 10, 24
    max_width = max(
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines
    )
    x0, y0 = 8, 8
    x1 = x0 + max_width + 2 * pad
    y1 = y0 + len(lines) * line_height + pad

    y1 = min(y1, image.shape[0])
    x1 = min(x1, image.shape[1])
    roi = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0)

    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x0 + pad, y0 + pad + (i + 1) * line_height - 7),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def print_result(result: vision.HandLandmarkerResult) -> None:
    """打印左右手信息与食指指尖坐标。"""
    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        category = result.handedness[idx][0]
        print(f"Handedness: {category.category_name} (score={category.score:.2f})")
        tip = hand_landmarks[INDEX_FINGER_TIP]
        print(
            f"Index finger tip coordinates: "
            f"({tip.x:.4f}, {tip.y:.4f})  # 归一化坐标"
        )
        if idx < len(result.hand_world_landmarks):
            world_tip = result.hand_world_landmarks[idx][INDEX_FINGER_TIP]
            print(
                f"Index finger tip world coordinates: "
                f"({world_tip.x:.3f}, {world_tip.y:.3f}, {world_tip.z:.3f})"
            )


def run_on_images(image_files=None) -> None:
    """静态图片模式：RunningMode.IMAGE + detect()。"""
    if image_files is None:
        image_files = IMAGE_FILES
    if not image_files:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with create_landmarker(vision.RunningMode.IMAGE) as landmarker:
        for idx, file in enumerate(image_files):
            image = cv2.imread(file)
            if image is None:
                print(f"无法读取图片: {file}")
                continue

            # 水平翻转后再检测，以得到正确的左右手（handedness）结果
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            result = landmarker.detect(mp_image)
            print(f"\n--- {file} ---")
            print_result(result)

            annotated_image = image.copy()
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(annotated_image, hand_landmarks)

            out_path = os.path.join(OUTPUT_DIR, f"annotated_image_{idx}.png")
            cv2.imwrite(out_path, cv2.flip(annotated_image, 1))
            print(f"结果已保存: {out_path}")


def run_on_webcam(camera_id: int = 0) -> None:
    """摄像头模式：RunningMode.VIDEO + detect_for_video()。"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头: {camera_id}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_index = 0
    smoothed_distance = {}  # 按手别缓存距离，用于显示平滑
    try:
        with create_landmarker(vision.RunningMode.VIDEO) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("忽略空帧。")
                    continue

                # Video 模式要求时间戳（毫秒）严格单调递增
                frame_index += 1
                timestamp_ms = int(frame_index * 1000 / fps)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                image_height, image_width, _ = frame.shape
                focal_px = focal_length_px(image_width)
                lines = ["ESC / q : quit"]

                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    draw_landmarks(frame, hand_landmarks)

                    name = (
                        result.handedness[idx][0].category_name
                        if idx < len(result.handedness)
                        else f"Hand{idx}"
                    )

                    # 手到相机的距离（米），带指数平滑
                    distance = estimate_distance_m(
                        hand_landmarks, image_width, image_height, focal_px
                    )
                    if distance is not None:
                        previous = smoothed_distance.get(name)
                        distance = (
                            distance
                            if previous is None
                            else previous + DIST_SMOOTHING * (distance - previous)
                        )
                        smoothed_distance[name] = distance
                        lines.append(f"{name}  dist: {distance:.2f} m")
                    else:
                        lines.append(f"{name}  dist: --")

                    # 世界坐标（米，原点在手掌几何中心）
                    if idx < len(result.hand_world_landmarks):
                        wrist = result.hand_world_landmarks[idx][0]
                        lines.append(
                            f"  wrist(m): x={wrist.x:+.3f} "
                            f"y={wrist.y:+.3f} z={wrist.z:+.3f}"
                        )

                # 文字要画在镜像显示之后，否则会被水平翻转
                display = cv2.flip(frame, 1)
                draw_info_panel(display, lines)
                cv2.imshow("MediaPipe Hands", display)

                key = cv2.waitKey(5) & 0xFF
                if key == 27 or key == ord("q"):  # ESC 或 q 退出
                    break
    except KeyboardInterrupt:
        print("已手动中断。")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_on_images()
        run_on_webcam()
    except KeyboardInterrupt:
        pass
