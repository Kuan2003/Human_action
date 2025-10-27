# src/datasets.py
from pathlib import Path
import torch,json,numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
from preprocess_ultils import *
from path_utils import SmartPathResolver

class HOISequenceDataset(Dataset):
    """
    Create temporal sequences of frames with all features ready for model input.
    Each item = {obj_imgs [T,C,H,W], pose_nodes [T,21,3], kin_feats [T,4], label, session_id}
     
    """
    def __init__(self, dataset_json, seq_len=16, stride=8,intrinsics=None, transform=None, drop_no_pose=True):
        self.seq_len=seq_len
        self.stride=stride
        self.intrinsics = intrinsics or {"fx": 611, "fy": 613, "cx": 313, "cy": 231}
        self.transform = transform or transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128,128)),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ])
        
        # Smart path resolution using utility
        self.path_resolver = SmartPathResolver(anchor_file=dataset_json)
        print(f"[Dataset] Detected project root: {self.path_resolver.get_project_root()}")
            
        with open(dataset_json,'r') as f:
            items=json.load(f)
            
        #group frames by session
        sessions ={}
        for it in items:
            if drop_no_pose and (not it.get("has_pose",True)):
                continue
            # Smart path resolution for all file types
            for path_key in ["rgb_path", "depth_path", "pose_path", "mask_path"]:
                if path_key in it:
                    it[path_key] = self.path_resolver.resolve(it[path_key], must_exist=False)
                continue
            s=it.get("session","default")
            sessions.setdefault(s,[]).append(it)
            
        #create sliding windows
        self.windows=[]
        for s,frames in sessions.items():
            frames=sorted(frames,key=lambda x:x["timestamp"])
            for i in range(0,max(1,len(frames)-seq_len+1),stride):
                win=frames[i:i+seq_len]
                if len(win)==seq_len:
                    self.windows.append((s,win))
        print(f"[Dataset] Loaded {len(self.windows)} sequences from {len(sessions)} sessions.")
        
        #build label index mapping
        labels=sorted({it["action"] for it in items})
        self.label2id={l:i for i,l in enumerate(labels)}
        self.id2label={i:l for l,i in self.label2id.items()}
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self,idx):
        session_id, frames=self.windows[idx]
        T=len(frames)
        obj_imgs,pose_nodes,kin_feats, labels=[],[],[],[]
        
        for f in frames:
            rgb=load_rgb(f["rgb_path"])
            depth=load_depth(f["depth_path"])
            mask=load_mask(f["mask_path"])
            pose2d=load_pose_json(f["pose_path"])
            pose3d=keypoint_2d_to_3d(pose2d,depth,self.intrinsics)
            obj_centroid=compute_centroid_3d(mask,depth,self.intrinsics)
            kin=compute_kin_features(obj_centroid,pose3d)
            
            ys,xs=np.where(mask>0)
            if len(xs)>0:
                x0,y0,x1,y1=xs.min(),ys.min(),xs.max(),ys.max()
                obj_crop=rgb[y0:y1+1,x0:x1+1]
            else:
                obj_crop=rgb
            obj_crop=cv2.resize(obj_crop,(128,128))
            obj_tensor=self.transform(obj_crop)
            obj_imgs.append(obj_tensor)
            pose_nodes.append(torch.tensor(pose3d,dtype=torch.float32))
            kin_feats.append(torch.tensor(kin,dtype=torch.float32))
            lab=f.get("action","None")
            labels.append(self.label2id.get(lab,self.label2id["None"]))
        return {
            "obj_imgs":torch.stack(obj_imgs,dim=0),    #[T,C,H,W]
            "pose_nodes":torch.stack(pose_nodes,dim=0), #[T,21,3]
            "kin_feats":torch.stack(kin_feats,dim=0), #[T,4]
            "label":torch.tensor(labels,dtype=torch.long), #[T]
            "session_id":session_id
        }

# Test code
if __name__=="__main__":
    ds=HOISequenceDataset("../../processed_data_auto/dataset.json",seq_len=8,stride=4)
    print(f"Dataset size: {len(ds)}")
    sample=ds[0]
    print("Sample keys:",sample.keys())
    print("obj_imgs shape:",sample["obj_imgs"].shape)
    print("pose_nodes shape:",sample["pose_nodes"].shape)
    print("kin_feats shape:",sample["kin_feats"].shape)
    print("label shape:",sample["label"].shape)
    print("Một vài giá trị label:", sample["label"])
    for i in range(20):
        s = ds[i]
        uniq = s["label"].unique()
        print(f"Seq {i} unique labels: {uniq} -> {[ds.id2label[int(x)] for x in uniq]}")
    
    def _find_project_root(self, dataset_json_path):
        """
        Smart project root detection - works across different environments
        """
        current_dir = os.path.dirname(dataset_json_path)
        
        # Method 1: Look for common project markers
        project_markers = [
            'processed_data_auto',    # Our processed data folder
            'recorded_dataset',       # Raw data folder  
            'Human_action',          # Main project folder
            '.git',                  # Git repository
            'requirements.txt',      # Python project
            'README.md'              # Project documentation
        ]
        
        # Traverse up to find project root
        max_levels = 5
        for _ in range(max_levels):
            # Check if current directory has project markers
            dir_contents = os.listdir(current_dir)
            if any(marker in dir_contents for marker in project_markers):
                # If processed_data_auto is in current dir, this is likely project root
                if 'processed_data_auto' in dir_contents:
                    return current_dir
                # If we're inside processed_data_auto, go up one level
                if os.path.basename(current_dir) == 'processed_data_auto':
                    return os.path.dirname(current_dir)
            
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached filesystem root
                break
            current_dir = parent_dir
        
        # Method 2: Fallback - use dataset.json location
        dataset_dir = os.path.dirname(dataset_json_path)
        if os.path.basename(dataset_dir) == 'processed_data_auto':
            return os.path.dirname(dataset_dir)
        
        # Method 3: Last resort - use current working directory
        cwd = os.getcwd()
        if 'Human_action' in cwd or 'Thuc_tap' in cwd:
            # Find the main project directory
            parts = cwd.split(os.sep)
            for i, part in enumerate(parts):
                if part in ['Human_action', 'Thuc_tap']:
                    return os.sep.join(parts[:i+2])  # Include one level above
        
        return os.path.dirname(dataset_json_path)
    
    def _resolve_path(self, file_path):
        """
        Smart path resolution - handles multiple scenarios
        """
        if not file_path:
            return file_path
            
        # If already absolute and exists, use it
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path
            
        # Try relative to project root
        candidate1 = os.path.join(self.base_dir, file_path)
        if os.path.exists(candidate1):
            return candidate1
            
        # Try relative to dataset json location
        candidate2 = os.path.join(os.path.dirname(self.dataset_json_path), file_path)
        if os.path.exists(candidate2):
            return candidate2
            
        # Try relative to current working directory
        candidate3 = os.path.join(os.getcwd(), file_path)
        if os.path.exists(candidate3):
            return candidate3
            
        # Try common project subdirectories
        common_subdirs = ['', 'processed_data_auto', 'Human_action/Source']
        for subdir in common_subdirs:
            candidate = os.path.join(self.base_dir, subdir, file_path)
            if os.path.exists(candidate):
                return candidate
                
        # If file starts with processed_data_auto/, try without it
        if file_path.startswith('processed_data_auto/'):
            rel_path = file_path[len('processed_data_auto/'):]
            processed_dir = os.path.join(self.base_dir, 'processed_data_auto')
            candidate = os.path.join(processed_dir, rel_path)
            if os.path.exists(candidate):
                return candidate
        
        # Return original path (might fail later, but at least we tried)
        print(f"[Warning] Could not resolve path: {file_path}")
        return file_path
            
if __name__=="__main__":
    ds=HOISequenceDataset("/home/kuan/Work_Space/Thuc_tap/processed_data_auto/dataset.json",seq_len=8,stride=4)
    print(f"Total sequences: {len(ds)}")
    print("Label mapping:", ds.label2id)
    print("Key sample:",ds[0].keys())
    sample=ds[0]
    print({k:v.shape if torch.is_tensor(v) else v for k,v in sample.items()})
    print("obj_imgs:", sample["obj_imgs"].shape)
    print("pose_nodes:", sample["pose_nodes"].shape)
    print("kin_feats:", sample["kin_feats"].shape)
    print("labels:", sample["label"].shape)
    print("Một vài giá trị label:", sample["label"])
    for i in range(20):
        s = ds[i]
        uniq = s["label"].unique()
        print(f"Seq {i} unique labels: {uniq} -> {[ds.id2label[int(x)] for x in uniq]}")