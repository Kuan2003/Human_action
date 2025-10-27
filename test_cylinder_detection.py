from ultralytics import YOLO
import cv2
import numpy as np

print("🔄 Loading cylinder detection model...")
model = YOLO("/home/kuan/Work_Space/Thuc_tap/Human_action/cylinder.pt")
print("✅ Model loaded! Detecting: cylinder")

def test_webcam():
    """Test model với webcam real-time"""
    cap = cv2.VideoCapture(0)  # Webcam
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("📹 Testing với webcam - Đặt một cylinder vào khung hình!")
    print("Press 'q' to quit, 'space' to save detection")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict với confidence thấp
        results = model.predict(frame, conf=0.1, verbose=False)
        
        # Vẽ kết quả lên frame
        annotated_frame = frame.copy()
        detection_found = False
        
        for r in results:
            if len(r.boxes) > 0:
                detection_found = True
                annotated_frame = r.plot()
                
                # In thông tin detection
                for i, (cls, conf) in enumerate(zip(r.boxes.cls, r.boxes.conf)):
                    print(f"Frame {frame_count}: Cylinder detected! Confidence: {conf:.2f}")
        
        # Thêm text status
        status_text = "🟢 CYLINDER DETECTED!" if detection_found else "🔴 Looking for cylinder..."
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if detection_found else (0, 0, 255), 2)
        
        cv2.imshow("Cylinder Detection - Real-time", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and detection_found:
            # Save detection
            filename = f"cylinder_detection_{frame_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"💾 Saved: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_sample_cylinder_image():
    """Tạo ảnh test cylinder đơn giản"""
    # Tạo ảnh với hình trụ đơn giản
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Background xám
    
    # Vẽ hình trụ (ellipse)
    center = (320, 240)
    axes = (100, 150)  # width, height
    cv2.ellipse(test_img, center, axes, 0, 0, 360, (200, 200, 200), -1)
    
    # Thêm shadow/gradient
    cv2.ellipse(test_img, (center[0]+10, center[1]+10), axes, 0, 0, 360, (150, 150, 150), -1)
    
    cv2.imwrite("test_cylinder.jpg", test_img)
    print("📸 Created test cylinder image: test_cylinder.jpg")
    
    # Test với ảnh này
    results = model.predict("test_cylinder.jpg", conf=0.1, verbose=True)
    
    for r in results:
        if len(r.boxes) > 0:
            print(f"✅ Detected {len(r.boxes)} cylinders in synthetic image!")
            annotated = r.plot()
            cv2.imshow("Test Cylinder Detection", annotated)
            cv2.imshow("Original Test Image", test_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ No cylinder detected in synthetic image")
            cv2.imshow("Test Image", test_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n🎯 Cylinder Detection Test")
    print("1. Webcam test (recommended)")
    print("2. Synthetic image test")
    
    choice = input("Choose option (1/2): ").strip()
    
    if choice == "1":
        test_webcam()
    elif choice == "2":
        test_with_sample_cylinder_image()
    else:
        print("Testing with webcam by default...")
        test_webcam()