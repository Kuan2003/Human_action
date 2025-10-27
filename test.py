import pyrealsense2 as rs
import numpy as np
import cv2
import torch

# Tạo pipeline cho hai camera
pipelines = [rs.pipeline() for _ in range(2)]
configs = [rs.config() for _ in range(2)]

# Thay bằng serial numbers thực tế
serials = ["213622078112", "832112070255"]  # Cập nhật với serial numbers của bạn

# Cấu hình stream và bật IR emitter
for i in range(2):
    configs[i].enable_device(serials[i])
    configs[i].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    configs[i].enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Bộ lọc Depth
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# Start streaming
for i in range(2):
    profile = pipelines[i].start(configs[i])
    # Tắt auto-exposure và bật IR emitter
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)  # Bật IR emitter
    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)  # Tắt auto-exposure
    depth_sensor.set_option(rs.option.exposure, 100)  # Điều chỉnh exposure
    depth_sensor.set_option(rs.option.gain, 16)  # Điều chỉnh gain

try:
    while True:
        for i in range(2):
            frames = pipelines[i].wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Áp dụng bộ lọc cho Depth
            depth_frame = decimation.process(depth_frame)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            # Convert sang numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Áp dụng colormap cho Depth
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Convert sang PyTorch tensor (nếu cần)
            depth_tensor = torch.from_numpy(depth_image).float().cuda()
            color_tensor = torch.from_numpy(color_image).float().cuda()

            # Hiển thị
            cv2.imshow(f'Camera {i+1} - RGB', color_image)
            cv2.imshow(f'Camera {i+1} - Depth', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()