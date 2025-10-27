import pyrealsense2 as rs
import numpy as np
import cv2
from glob import glob
import os

#Checkboard (10x7 , 9x6 inner corners, size 0.025m)
CHECKERBOARD=(9,6)
square_size=0.025

#Termination criteria
criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

#Serial numbers of the camera
serials=""

#Create folder to save calibration data
output_dir="calibration_images"
os.makedirs(output_dir,exist_ok=True)
img_count=0


# Create pipeline
pipeline=rs.pipeline()
config=rs.config()
config.enable_device(serials)
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)

# Start streaming
pipeline.start(config)
print("Press 'c' to capture calibration image pair, 'q' to quit.")
try:
    while True:
        frames=pipeline.wait_for_frames()
        color_frame=frames.get_color_frame()
        if not color_frame:
            continue
        color_image= np.assanyarray(color_frame.get_data())
        cv2.imshow("RGB Stream",color_image)

        key=   cv2.waitKey(1) & 0xFF
        if key==ord('c'):
            #save color image
            img_name=os.path.join(output_dir,f"img_{img_count:03d}.jpg")
            cv2.imwrite(img_name,color_image)
            print(f"Saved :{img_name}")
            img_count+=1
        
            #check corners
            gray=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
            ret, corners=cv2.findChessboardCorners(gray,CHECKERBOARD,None)
            if ret==True:
                cv2.drawChessboardCorners(color_image,CHECKERBOARD,corners,ret)
                cv2.imshow("Detected Corners",color_image)
                cv2.waitKey(500)
            else:
                print("Chessboard corners not found, try again.")
        elif key==ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()


# Calibration process
objpoints=[] #3d point in real world space
imgpoints=[] #2d points in image plane

objp=np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3),np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp*=square_size

images=glob(f"{output_dir}/*.jpg")
for fname in images:
    img=cv2.imread(fname)
    gray=cv2.cvtColor(img,cv2.COLOR_BRG2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,CHECKERBOARD,None)
    if ret==True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        print(f"Processed {fname}")
if len(objpoints)>0 :
    #calibrate camera
    ret,matrix,dist,rvecs,tvecs=cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

    #save calibration data
    np.savez("camera_calibration_data.npz",matrix=matrix,dist=dist,rvecs=rvecs,tvecs=tvecs)
    print("Calibration successful. Data saved to camera_calibration_data.npz")
    print("Camera matrix:\n",matrix)
    print("Distortion coefficients:\n",dist)

else:
    print("No chessboard corners found in any image. Calibration failed.")