# src/preprocess_ultils.py
import numpy as np,cv2, json
from pathlib import Path

def load_mask(mask_path):
    """Load object mask (binary or uint8)"""
    if mask_path.endswith(".npy"):
        mask=np.load(mask_path)
    else:
        mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    mask=(mask>0).astype(np.uint8)
    return mask

def load_depth(depth_path):
    """Load depth map (16-bit PNG or .npy)"""
    if depth_path.endswith(".npy"):
        return np.load(depth_path)
    else:
        depth=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        #convert to meters
        depth=depth.astype(np.float32)/1000.0
        return depth    
    

def load_rgb(rgb_path):
    """Load RGB image and convert to RGB order"""
    img=cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rgb_path}")
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR to RGB

def load_pose_json(pose_path):
    """Load 2D pose from JSON file"""
    with open(pose_path,'r') as f:
        data=json.load(f)
    if "pose_3d" in data:
        pts=data["pose_3d"]
        kps=np.array([[p[0],p[1],p[2]] for p in pts])
    # elif "hand_landmarks" in data:
    #     pts=data["hand_landmarks"]
    #     kps=np.array([[p[0],p[1],0.0] for p in pts])
    else:
        kps=np.zeros((21,3),dtype=np.float32)
    return kps #shape (21,3)

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path,'r') as f:
        intrinsic=json.load(f)
    intrinsic=np.array(intrinsic).reshape(3,3)
    return intrinsic

def object_crop_from_mask(rgb,mask,out_size=(128,128)):
    ys,xs=np.where(mask>0)
    if len(xs)==0:
        #fallback: center crop
        h,w=rgb.shape[:2]
        cx,cy=w//2,h//2
        x0=max(0,cx-out_size[1]//2)
        y0=max(0,cy-out_size[0]//2)
        x1=min(w,cx+out_size[1]//2)
        y1=min(h,cy+out_size[0]//2)
    else:
        x0,y0=xs.min(),ys.min()
        x1,y1=xs.max(),ys.max()
        
    crop= rgb[y0:y1+1, x0:x1+1]
    crop=cv2.resize(crop,out_size) 
    return crop, (x0,y0,x1,y1)

def compute_centroid_3d(mask,depth,intrinsics):
    """Compute 3d centroid from mask & depth"""
    ys,xs=np.where(mask>0)
    if len(xs)==0:
        return np.zeros(3)
    
    z=np.median(depth[ys,xs])
    if np.isnan(z) or z<=0:
        z=np.nanmedian(depth)
    cx, cy= np.mean(xs),np.mean(ys)
    
    #intrinsics: fx,fy,cx0,cy0 -> backproject
    X=(cx-intrinsics['cx']) * z / intrinsics['fx']
    Y=(cy-intrinsics['cy']) * z / intrinsics['fy']
    
    return np.array([X,Y,z],dtype=np.float32)

def keypoint_2d_to_3d(kps2d, depth, intrinsics):
    """Prohect 2D keypoints (u,v) to 3D ussing depth map"""
    kps3d=[]
    h,w=depth.shape
    fx,fy,cx0,cy0=intrinsics['fx'],intrinsics['fy'],intrinsics['cx'],intrinsics['cy']
    for (x,y,z2d) in kps2d:
        u=int(np.clip(x*w,0,w-1))
        v=int(np.clip(y*h,0,h-1))
        z=float(depth[v,u])
        if z <=0 or np.isnan(z):
            z=np.nanmedian(depth)
            
        X=(u-cx0) *z / fx
        Y=(v-cy0) *z / fy
        kps3d.append([X,Y,z])
    return np.array(kps3d,dtype=np.float32)

def compute_kin_features(obj_centroid,hand_kps3d):
    """Compute kinematic features:distance, relative vector, centroid velocity placeholder"""
    hand_centroid=np.mean(hand_kps3d,axis=0)
    vec=hand_centroid-obj_centroid
    dist=np.linalg.norm(vec)
    kin=np.concatenate(([dist],vec))
    return kin.astype(np.float32)


