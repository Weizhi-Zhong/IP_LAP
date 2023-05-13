import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False,act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        if act=='ReLU':
            self.act = nn.ReLU()
        elif act=='Tanh':
            self.act =nn.Tanh()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()

        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Fusion_transformer_encoder(nn.Module):
    def __init__(self,T, d_model, nlayers, nhead, dim_feedforward,  # 1024   128
                 dropout=0.1):
        super().__init__()
        self.T=T
        self.position_v = PositionalEmbedding(d_model=512)  #for visual landmarks
        self.position_a = PositionalEmbedding(d_model=512)  #for audio embedding
        self.modality = nn.Embedding(4, 512, padding_idx=0)  # 1 for pose,  2  for  audio, 3 for reference landmarks
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self,ref_embedding,mel_embedding,pose_embedding):#(B,Nl,512)  (B,T,512)    (B,T,512)

        # (1).  positional(temporal) encoding
        position_v_encoding = self.position_v(pose_embedding)  # (1,  T, 512)
        position_a_encoding = self.position_a(mel_embedding)

        #(2)  modality encoding
        modality_v = self.modality(1 * torch.ones((pose_embedding.size(0), self.T), dtype=torch.int).cuda())
        modality_a = self.modality(2 * torch.ones((mel_embedding.size(0),  self.T), dtype=torch.int).cuda())

        pose_tokens = pose_embedding + position_v_encoding + modality_v    #(B , T, 512 )
        audio_tokens = mel_embedding + position_a_encoding + modality_a    #(B , T, 512 )
        ref_tokens = ref_embedding + self.modality(
            3 * torch.ones((ref_embedding.size(0), ref_embedding.size(1)), dtype=torch.int).cuda())

        #(3) concat tokens
        input_tokens = torch.cat((ref_tokens, audio_tokens, pose_tokens), dim=1)  # (B, 1+T+T, 512 )
        input_tokens = self.dropout(input_tokens)

        #(4) input to transformer
        output = self.transformer_encoder(input_tokens)
        return output


class Landmark_generator(nn.Module):
    def __init__(self,T,d_model,nlayers,nhead,dim_feedforward,dropout=0.1):
        super(Landmark_generator, self).__init__()
        self.mel_encoder=nn.Sequential(  #  (B*T,1,hv,wv)
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0,act='Tanh'),
            )

        self.ref_encoder=nn.Sequential( # (B*Nl,2,131)
            Conv1d(2, 4, 3, 1, 1),  #131

            Conv1d(4, 8, 3, 2,1), #66
            Conv1d(8, 8, 3, 1, 1,residual=True),
            Conv1d(8, 8, 3, 1, 1,residual=True),

            Conv1d(8, 16, 3, 2, 1),  # 33
            Conv1d(16, 16, 3, 1, 1, residual=True),
            Conv1d(16, 16, 3, 1, 1, residual=True),

            Conv1d(16, 32, 3, 2,1),#   17
            Conv1d(32, 32, 3, 1, 1,residual=True),
            Conv1d(32, 32, 3, 1, 1,residual=True),

            Conv1d(32, 64, 3, 2,1), #   9
            Conv1d(64, 64, 3, 1, 1,residual=True),
            Conv1d(64, 64, 3, 1, 1,residual=True),

            Conv1d(64, 128, 3, 2,1), #   5
            Conv1d(128, 128, 3, 1, 1,residual=True),
            Conv1d(128, 128, 3, 1, 1,residual=True),

            Conv1d(128, 256, 3, 2,1), #3
            Conv1d(256, 256, 3, 1, 1,residual=True),

            Conv1d(256, 512, 3, 1,0), #1
            Conv1d(512, 512, 1, 1,0,act='Tanh'), #1
        )
        self.pose_encoder=nn.Sequential(   # (B*T,2,74)
            Conv1d(2, 4, 3, 1, 1),

            Conv1d(4, 8, 3, 1, 1),  #74
            Conv1d(8, 8, 3, 1, 1,residual=True),
            Conv1d(8, 8, 3, 1, 1, residual=True),

            Conv1d(8, 16, 3, 2, 1),  # 37
            Conv1d(16, 16, 3, 1, 1, residual=True),
            Conv1d(16, 16, 3, 1, 1, residual=True),

            Conv1d(16, 32, 3, 2, 1), # 19
            Conv1d(32, 32, 3, 1, 1, residual=True),
            Conv1d(32, 32, 3, 1, 1, residual=True),

            Conv1d(32, 64, 3, 2, 1), #10
            Conv1d(64, 64, 3, 1, 1, residual=True),
            Conv1d(64, 64, 3, 1, 1, residual=True),

            Conv1d(64, 128, 3, 2, 1), # 5
            Conv1d(128, 128, 3, 1, 1, residual=True),
            Conv1d(128, 128, 3, 1, 1, residual=True),

            Conv1d(128, 256, 3, 2, 1),  # 3
            Conv1d(256, 256, 3, 1, 1, residual=True),
            Conv1d(256, 256, 3, 1, 1, residual=True),

            Conv1d(256, 512, 3, 1, 0),  # 1
            Conv1d(512, 512, 1, 1, 0, residual=True,act='Tanh'),
        )

        self.fusion_transformer = Fusion_transformer_encoder(T,d_model,nlayers,nhead,dim_feedforward,dropout)

        self.mouse_keypoint_map = nn.Linear(d_model, 40 * 2)
        self.jaw_keypoint_map = nn.Linear(d_model, 17 * 2)

        self.apply(weight_init)
        self.Norm=nn.LayerNorm(512)

    def forward(self, T_mels, T_pose, Nl_pose, Nl_content):
                # (B,T,1,hv,wv) (B,T,2,74) (B,N_l,2,74)  (B,N_l,2,57)
        B,T,N_l= T_mels.size(0),T_mels.size(1),Nl_content.size(1)

        #1. obtain full reference landmarks
        Nl_ref = torch.cat([Nl_pose, Nl_content], dim=3)  #(B,Nl,2,131=74+57)
        Nl_ref = torch.cat([Nl_ref[i] for i in range(Nl_ref.size(0))], dim=0)  # (B*Nl,2,131)

        T_mels=torch.cat([T_mels[i] for i in range(T_mels.size(0))],dim=0) #(B*T,1,hv,wv)
        T_pose = torch.cat([T_pose[i] for i in range(T_pose.size(0))],dim=0)  # (B*T,2,74)

        # 2. get embedding
        mel_embedding=self.mel_encoder(T_mels).squeeze(-1).squeeze(-1)#(B*T,512)
        pose_embedding=self.pose_encoder(T_pose).squeeze(-1)  # (B*T,512)
        ref_embedding = self.ref_encoder(Nl_ref).squeeze(-1)  # (B*Nl,512)
        #normalization
        mel_embedding = self.Norm(mel_embedding)  # (B*T,512)
        pose_embedding =self.Norm(pose_embedding)   # (B*T,512)
        ref_embedding =  self.Norm(ref_embedding) # (B*Nl,512)

        mel_embedding = torch.stack(torch.split(mel_embedding,T),dim=0) #(B,T,512)
        pose_embedding = torch.stack(torch.split(pose_embedding, T), dim=0) # (B,T,512)
        ref_embedding=torch.stack(torch.split(ref_embedding,N_l,dim=0),dim=0) #(B,N_l,512)

        #3. fuse embedding
        output_tokens=self.fusion_transformer(ref_embedding,mel_embedding,pose_embedding)

        #4.output  landmark
        lip_embedding=output_tokens[:,N_l:N_l+T,:] #(B,T,dim)
        jaw_embedding=output_tokens[:,N_l+T:,:] #(B,T,dim)
        output_mouse_landmark=self.mouse_keypoint_map(lip_embedding)  ##(B,T,40*2)
        output_jaw_landmark=self.jaw_keypoint_map(jaw_embedding)   ##(B,T,17*2)

        predict_content=torch.reshape(torch.cat([output_jaw_landmark,output_mouse_landmark],dim=2),(B,T,-1,2))   #(B,T,57,2)
        predict_content=torch.cat([predict_content[i] for i in range(predict_content.size(0))],dim=0).permute(0,2,1)#(B*T,2,57)
        return predict_content  #(B*T,2,57)

