import os, cv2, time, json, csv
import numpy as np
from datetime import datetime
import pyrealsense2 as rs

# Optional: pose extraction
USE_MEDIAPIPE = True
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    USE_MEDIAPIPE = False
    print("⚠️ MediaPipe not found, pose extraction disabled.")

# Config
OUTPUT_ROOT = "recorded_dataset"
FRAME_WIDTH, FRAME_HEIGHT, FPS = 640, 480, 15
DEPTH_MEDIAN_WIN = 5
DEMO_MODE = False  # Set True để chạy demo không cần camera thật

# -------------------------------
# Helper functions
# -------------------------------
def safe_mkdir(path): os.makedirs(path, exist_ok=True)

def sample_depth_median(depth_m, x, y, win=DEPTH_MEDIAN_WIN):
    H, W = depth_m.shape
    r = win // 2
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[~np.isnan(patch)]
    if valid.size == 0: return None
    return float(np.median(valid))

def setup_pipeline(width, height, fps):
    try:
        # Kiểm tra thiết bị RealSense có sẵn không
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("❌ Không tìm thấy thiết bị RealSense nào!")
        
        print(f"✅ Tìm thấy {len(devices)} thiết bị RealSense")
        for i, device in enumerate(devices):
            print(f"   Thiết bị {i}: {device.get_info(rs.camera_info.name)}")
        
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"✅ Camera đã được khởi tạo - Depth scale: {depth_scale}")
        return pipeline, align, depth_scale, profile
    except Exception as e:
        print(f"❌ Lỗi khởi tạo camera: {e}")
        raise

def get_intrinsics_json(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "fx": intr.fx,
        "fy": intr.fy,
        "model": str(intr.model).split('.')[-1],
        "coeffs": intr.coeffs
    }

# -------------------------------
# Recorder class
# -------------------------------
class Recorder:
    def __init__(self):
        if DEMO_MODE:
            print("🎭 Chạy ở chế độ DEMO (không cần camera)")
            self.camera_available = True
            self.depth_scale = 0.001  # Demo depth scale
        else:
            try:
                self.pipeline, self.align, self.depth_scale, self.profile = setup_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS)
                self.camera_available = True
            except Exception as e:
                print(f"❌ Không thể khởi tạo camera: {e}")
                self.camera_available = False
                return
            
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) if USE_MEDIAPIPE else None

        self.recording = False
        self.markers = []
        self.timestamps = []
        self.frame_idx = 0
        self.session_dir = None
        self.rgb_dir = None
        self.depth_dir = None
        self.pose_dir = None

    def start_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(OUTPUT_ROOT, f"session_{ts}")
        self.rgb_dir = os.path.join(self.session_dir, "rgb")
        self.depth_dir = os.path.join(self.session_dir, "depth")
        self.pose_dir = os.path.join(self.session_dir, "pose")

        for d in [OUTPUT_ROOT, self.session_dir, self.rgb_dir, self.depth_dir, self.pose_dir]:
            safe_mkdir(d)

        meta = {
            "session_name": f"session_{ts}",
            "timestamp_start": time.time(),
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": FPS,
            "depth_scale_m": self.depth_scale,
            "intrinsics": get_intrinsics_json(self.profile)
        }
        with open(os.path.join(self.session_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.recording = True
        self.frame_idx = 0
        self.markers.clear()
        self.timestamps.clear()
        print(f"🎬 Recording started: {self.session_dir}")

    def stop_session(self):
        self.recording = False
        with open(os.path.join(self.session_dir, "timestamps.json"), "w") as f:
            json.dump(self.timestamps, f, indent=2)
        with open(os.path.join(self.session_dir, "markers.json"), "w") as f:
            json.dump(self.markers, f, indent=2)
        csv_path = os.path.join(self.session_dir, "timestamps.csv")
        with open(csv_path, "w", newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=["frame_idx", "ts_system", "rgb_path", "depth_path"])
            writer.writeheader()
            for r in self.timestamps: writer.writerow(r)
        print(f"✅ Session saved: {self.session_dir}")

    def add_marker(self, label):
        t = time.time()
        self.markers.append({"timestamp": t, "label": label})
        print(f"📍 Marker: {label} @ {t:.3f}")

    def capture_once(self):
        if DEMO_MODE:
            # Tạo ảnh demo
            color_image = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            # Tạo gradient cho depth
            y, x = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
            depth_m = (x + y) / (FRAME_WIDTH + FRAME_HEIGHT) * 2.0  # 0-2 mét
            return color_image, depth_m, time.time()
        else:
            try:
                # Đợi frame với timeout ngắn hơn
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    return None, None, None

                color_image = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_m = depth_raw.astype(np.float32) * self.depth_scale
                depth_m[depth_raw == 0] = np.nan
                return color_image, depth_m, time.time()
            except RuntimeError as e:
                print(f"⚠️ Lỗi capture frame: {e}")
                return None, None, None

    def record_frame(self, color_image, depth_m, ts):
        rgb_path = os.path.join(self.rgb_dir, f"frame_{self.frame_idx:06d}.png")
        cv2.imwrite(rgb_path, color_image)

        depth_path = os.path.join(self.depth_dir, f"depth_{self.frame_idx:06d}.npy")
        np.save(depth_path, depth_m.astype(np.float32))

        # Pose extraction
        if self.pose_detector:
            img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(img_rgb)
            pose2d, pose3d = [], []
            if results.pose_landmarks:
                h, w, _ = color_image.shape
                for lm in results.pose_landmarks.landmark:
                    x_px = int(round(lm.x * w))
                    y_px = int(round(lm.y * h))
                    z_m = sample_depth_median(depth_m, x_px, y_px)
                    pose2d.append({"x": lm.x * w, "y": lm.y * h, "score": lm.visibility})
                    pose3d.append({"x": lm.x * w, "y": lm.y * h, "z_m": z_m, "score": lm.visibility})
            pose_path = os.path.join(self.pose_dir, f"pose_{self.frame_idx:06d}.json")
            with open(pose_path, "w") as pf:
                json.dump({"pose2d": pose2d, "pose3d": pose3d}, pf)

        self.timestamps.append({
            "frame_idx": self.frame_idx,
            "ts_system": ts,
            "rgb_path": rgb_path,
            "depth_path": depth_path
        })
        self.frame_idx += 1


# -------------------------------
# MAIN LOOP (Hiển thị RGB + Depth)
# -------------------------------
if __name__ == "__main__":
    # Set CV2 backend để tránh lỗi Qt
    import os
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    print("🔄 Khởi tạo recorder...")
    rec = Recorder()
    
    # Kiểm tra camera có sẵn không
    if not rec.camera_available:
        print("❌ Camera không khả dụng. Vui lòng kiểm tra kết nối RealSense.")
        exit(1)
    
    print("🎥 Action Recorder (RGB-D + Pose + Markers)")
    print("---------------------------------------------------")
    print("Controls:")
    print("  [s] Start / Stop recording")
    print("  [q] Quit")
    print("  [a] Approach   [g] Grasp   [t] Transport   [r] Release   [n] None")
    print("---------------------------------------------------")
    print("✅ Camera sẵn sàng! Đang hiển thị video...")

    try:
        while True:
            color_image, depth_m, ts = rec.capture_once()
            if color_image is None:
                # Nếu không có frame, chờ một chút rồi thử lại
                time.sleep(0.01)
                continue

            # Depth visualization
            depth_vis = np.nan_to_num(depth_m, nan=0.0)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_vis, alpha=255.0 / np.nanmax(depth_vis) if np.nanmax(depth_vis) > 0 else 1),
                cv2.COLORMAP_JET
            )

            # Nếu đang ghi, lưu frame
            if rec.recording:
                rec.record_frame(color_image, depth_m, ts)
                # Thêm text "REC" vào cả 2 ảnh
                cv2.putText(color_image, f"REC {rec.frame_idx}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(depth_colormap, f"REC {rec.frame_idx}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- Hiển thị RGB và Depth trong 2 cửa sổ riêng biệt ---
            cv2.imshow("RGB Camera", color_image)
            cv2.imshow("Depth Camera", depth_colormap)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if rec.recording:
                    rec.stop_session()
                print("👋 Exiting...")
                break

            elif key == ord('s'):
                if not rec.recording:
                    rec.start_session()
                else:
                    rec.stop_session()

            elif key == ord('a'):
                rec.add_marker("Approach")
            elif key == ord('g'):
                rec.add_marker("Grasp")
            elif key == ord('t'):
                rec.add_marker("Transport")
            elif key == ord('r'):
                rec.add_marker("Release")
            elif key == ord('n'):
                rec.add_marker("None")
                
    except KeyboardInterrupt:
        print("\n👋 Thoát bằng Ctrl+C")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        cv2.destroyAllWindows()
        if 'rec' in locals() and hasattr(rec, 'pipeline'):
            rec.pipeline.stop()
