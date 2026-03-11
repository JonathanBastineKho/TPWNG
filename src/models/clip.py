import torch
import torch.nn as nn
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

#Preliminary setup from paper: 

'''
Pretrained clip ver = "ViT-B/16"
Feature Dimension D = 512
Learnable Sequence Length in Text Prompt l = 8
'''

pretrained_clip_ver = "ViT-B/16"
feature_dimension = 512
learnable_seq_len = 8


#Load model and params
clip_model, preprocess = clip.load(pretrained_clip_ver, device=device)
clip_model = clip_model.float() #debug
embed_dim = feature_dimension
for param in clip_model.parameters():
    param.requires_grad = False

#Unfreeze one layer for training
clip_model.text_projection.requires_grad = True

class PromptTextEncoder(nn.Module):
    def __init__(self, classnames, clip_model = clip_model, embed_dim = embed_dim, n_ctx=learnable_seq_len):
        super().__init__()

        self.clip_model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        dtype = clip_model.dtype
        ctx_dim = embed_dim

        self.ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim, dtype=dtype))
        prompts = [name for name in classnames]
        tokenised = torch.cat([clip.tokenize(p) for p in prompts])

        self.register_buffer("tokenised_prompts", tokenised)

    def forward(self):
        tokenised = self.tokenised_prompts
        clip_model = self.clip_model

        #token embedding and transformation
        x = clip_model.token_embedding(tokenised).type(clip_model.dtype)
        ctx = self.ctx.unsqueeze(0).expand(x.shape[0], -1, -1)

        x = torch.cat(
            [
                x[:, :1, :],     #SOS
                ctx,             #ctx1..ctx8
                x[:, 1:, :]      #original tokens
            ],
            dim=1
        )



        x = x[:, :77, :]
        x = x + clip_model.positional_embedding[:x.shape[1]]

        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)

        x = clip_model.ln_final(x)
        eot_pos = tokenised.argmax(dim=-1)
        sentence_feature = x[torch.arange(x.shape[0]), eot_pos]
        text_features = sentence_feature @ clip_model.text_projection

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features