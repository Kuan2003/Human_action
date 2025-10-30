#src/models.py
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GCNConv, SAGEConv

#-----Object encoder (CNN) -----
class ObjectEncoder(nn.Module):
    def __init__(self,out_dim=128,pretrained=True):
        super(ObjectEncoder,self).__init__()
        backbone=models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone=backbone.features
        in_features=backbone.last_channel #1028
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Linear(1280,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,out_dim)
        )
        
    def forward (self,x):
        #x: [B,3,H,W]
        feat=self.backbone(x)
        feat=self.pool(feat).flatten(1)  #[B,1280]
        feat=self.fc(feat) #[B,out_dim]
        feat=F.normalize(feat,dim=-1)
        
        return feat #[B,out_dim]

#-----Pose encoder (GNN backbone -Pytorch Geometric) -----
class PoseGNN_PyG(nn.Module):
    def __init__(self, in_dim=3, hidden=128, out_dim=128, model_type="sage"):
        super().__init__()
        if model_type == "sage":
            # GraphSAGE requires num_layers parameter
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, out_dim)
        else:
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, out_dim)
        self.model_type = model_type

    def forward(self, x, edge_index):
        """
        x: [B, N, F] hoặc [N, F]
        edge_index: [2, E]
        """
        if x.ndim == 3:  # batch input
            out = []
            for i in range(x.shape[0]):
                h = self._forward_single(x[i], edge_index)  # [N, D]
                out.append(h.mean(dim=0))  # ✅ global mean pooling
            return torch.stack(out, dim=0)  # [B, D]
        else:
            h = self._forward_single(x, edge_index)
            return h.mean(dim=0)  # [D]

    def _forward_single(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
#-----Temporal model (LSTM) -----
class FusionLSTM(nn.Module):
    """
    Input: concatenated features [obj_feat | pose_feat | kin_feat] per frame
    Output: per-frame action logits
    """
    def __init__(self,input_dim=256,hidden=128,num_classes=5,num_layers=1,bidirectional=False):
        super(FusionLSTM,self).__init__()
        self.lstm=nn.LSTM(input_dim,hidden,num_layers=num_layers,
                          batch_first=True,bidirectional=bidirectional)
        self.drop=nn.Dropout(0.3)
        d=2 if bidirectional else 1
        self.fc=nn.Linear(hidden*d,num_classes)
    def forward(self,x):
        #x: [B,T,input_dim]
        out,_=self.lstm(x) #[B,T,hidden*d]
        out=self.drop(out)
        logits=self.fc(out) #[B,T,num_classes]
        return logits
class SimpleHead(nn.Module):
    """Phần phân loại đầu ra cho Stage A hoặc B"""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)
#----Wrapper (optional) -----
class HOIModel(nn.Module):
    """
    combine object encoder, pose encoder and temporal model
    finetune Stage D or inference realtime Stage E
    """
    def __init__(self,obj_dim=128,pose_dim=128,kin_dim=4,hidden=128,num_classes=5,pretrained_obj=True):
        super(HOIModel,self).__init__()
        self.obj_enc=ObjectEncoder(out_dim=obj_dim,pretrained=pretrained_obj)
        self.pose_enc=PoseGNN_PyG(in_dim=3,hidden=64,out_dim=pose_dim)
        self.temporal=FusionLSTM(input_dim=obj_dim+pose_dim+kin_dim,
                                 hidden=hidden,
                                 num_classes=num_classes)

        #pre_define hand skeleton edges (for inference)
        self.register_buffer("edges", torch.tensor([
            [0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20]
        ], dtype=torch.long).t())
        
    def forward(self,obj_imgs,pose_nodes,kin_feats):
        """
        obj_imgs: [B,T,3,H,W]
        pose_nodes: [B,T,21,3]
        kin_feats: [B,T,4]
        """
        
        B,T=obj_imgs.shape[:2]
        device=obj_imgs.device
        
        obj_feats, pose_feats=[],[]
        
        for t in range(T):
            #object feature
            img_t=obj_imgs[:,t] #[B,3,H,W]
            feat_obj=self.obj_enc(img_t) #[B,obj_dim]
            obj_feats.append(feat_obj)
            
            #pose features per batch element
            feats_t=[]
            for b in range(B):
                x=pose_nodes[b,t].to(device) #[21,3]
                emb=self.pose_enc(x,self.edges.to(device)) #[pose_dim]
                feats_t.append(emb)
            pose_feats.append(torch.stack(feats_t,dim=0)) #[B,pose_dim]
        
        obj_feats=torch.stack(obj_feats,dim=1) #[B,T,obj_dim]
        pose_feats=torch.stack(pose_feats,dim=1) #[B,T,pose_dim]
        fused=torch.cat([obj_feats,pose_feats,kin_feats],dim=-1) #[B,T,obj+pose+kin]
        logits=self.temporal(fused) #[B,T,num_classes]
        return logits  #[B,T,num_classes]
    
def get_hand_edges():
    """
    Trả về danh sách cạnh bàn tay (edge_index) dùng cho pose GNN.
    """
    return torch.tensor([
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20]
    ], dtype=torch.long).t()