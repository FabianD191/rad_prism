# model.py
# ============================================================================
# Vision -> Concept model.
#
# A vision backbone encodes the chest X-ray into a token sequence. K learnable
# "concept tokens" then attend to those vision tokens (cross-attention) to
# produce one image-derived embedding per clinical concept. During pretraining
# these concept embeddings are aligned with the text embeddings of the matching
# report snippets (InfoNCE). Optional per-concept binary classification heads
# can be enabled for the downstream fine-tuning stage.
#
# Requires PyTorch >= 2.0 and torchvision >= 0.15.
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os


# -------------------------
# Config
# -------------------------

@dataclass
class ModelConfig:
    # Concepts
    concept_names: List[str]  # fixed order of concepts used throughout
    cls_concept_names: Optional[List[str]] = None  # optional, if classification concepts differ
    cls_head_map: Optional[List[int]] = None  # maps each cls concept -> index of concept token
    use_cls_heads: bool = True
    d_model: int = 512        # internal feature dim for attention + heads
    n_heads: int = 8          # cross-attention heads
    dropout: float = 0.1

    # Vision backbone: "resnet50" | "dinov2" | "rad_dino_maira_2"
    vision_backbone: str = "rad_dino_maira_2"
    pretrained_vision: bool = False           # only use with internet / auto-download allowed
    vision_weights_path: Optional[str] = None  # path to a local .pth; overrides pretrained_vision
    # DINOv2 (local torch.hub): path to the unzipped repo (folder with hubconf.py)
    dinov2_repo_path: Optional[str] = None
    # DINOv2: path to the .pth weights file
    dinov2_weights_path: Optional[str] = None
    # RAD-DINO-MAIRA-2 (HuggingFace): local model directory (config.json + weights)
    rad_dino_model_dir: Optional[str] = None

    # Text projections
    text_in_dim: int = 768    # dimension of the pre-computed text embeddings
    project_text: bool = True # project text to d_model

    # Convenience for spatial maps
    # For ResNet we infer token grid from feature map. For ViT we compute from input_size/patch
    record_attn_maps: bool = True

    # Init
    concept_token_init_scale: float = 0.02

    # Inference
    return_concept_embeddings: bool = True


# -------------------------
# Utilities
# -------------------------

def l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def _maybe_project(x: torch.Tensor, proj: Optional[nn.Module]) -> torch.Tensor:
    return proj(x) if proj is not None else x


# -------------------------
# Vision Backbones
# -------------------------

class ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone that outputs a spatial feature map and a token sequence.
    """
    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None, out_dim: int = 2048):
        super().__init__()
        if weights_path is not None:
            # Local weights
            net = models.resnet50(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
        else:
            # Standard torchvision mechanism (only meaningful with internet access)
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
        # keep up to layer4
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_dim = out_dim  # 2048 for resnet50

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x: [B,3,H,W]
        returns:
          fmap: [B, C, H', W']  with stride 32
          H', W' are spatial dims for attention heatmaps
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fmap = self.layer4(x)  # [B, 2048, H', W']
        return fmap, fmap.shape[-2], fmap.shape[-1]


class ViTBackbone(nn.Module):
    """
    DINOv2 ViT-Base/14 wrapper (Offline / Local Version).
    Loads model definition from a local clone of the repo and weights from a local .pth file.
    """
    def __init__(self, repo_path: str, weights_path: str):
        super().__init__()
        
        # 1. Validation
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"DINOv2 repo not found at: {repo_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"DINOv2 weights not found at: {weights_path}")

        print(f"Loading DINOv2 Code from: {repo_path}")
        print(f"Loading DINOv2 Weights from: {weights_path}")

        # 2. Load Model Architecture Locally
        # source='local' forces torch.hub to look at the folder, not the internet
        self.net = torch.hub.load(repo_path, 'dinov2_vitb14', source='local', pretrained=False)
        
        # 3. Load Weights Manually
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state_dict)

        # 4. Set Dimensions
        self.hidden_dim = 768
        self.patch_size = 14  # DINOv2 uses 14px patches

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        x: [B,3,H,W]
        Returns:
          tokens: [B, N, 768]
          cls:    [B, 768]
          gh, gw: grid height/width
        """
        # DINOv2 forward_features returns a dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
        ret = self.net.forward_features(x)
        
        cls_token = ret['x_norm_clstoken']       # [B, 768]
        patch_tokens = ret['x_norm_patchtokens'] # [B, N, 768]
        
        # Calculate grid size dynamically
        gh = x.shape[-2] // self.patch_size
        gw = x.shape[-1] // self.patch_size
        
        return patch_tokens, cls_token, gh, gw



class HFDinov2Backbone(nn.Module):
    """
    HuggingFace Transformers Dinov2Model wrapper (offline/local).
    Expects a local directory containing config.json + model.safetensors (+ preprocessor_config.json if you also want processor).
    """
    def __init__(self, model_dir: str):
        super().__init__()
        from transformers import AutoModel

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"HF model dir not found at: {model_dir}")

        # local_files_only avoids any network calls
        self.net = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.net.eval()  # you can override in training if you fine-tune

        # config has hidden_size=768, patch_size=14, image_size=518 for rad-dino-maira-2
        self.hidden_dim = int(self.net.config.hidden_size)
        self.patch_size = int(self.net.config.patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # transformers Dinov2Model expects pixel_values=
        out = self.net(pixel_values=x, return_dict=True)

        # last_hidden_state: [B, 1+N, D] (CLS + patches)
        tok = out.last_hidden_state
        cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else tok[:, 0]
        patches = tok[:, 1:]

        # robust grid computation (works for non-square too)
        gh = x.shape[-2] // self.patch_size
        gw = x.shape[-1] // self.patch_size
        return patches, cls, gh, gw



# -------------------------
# Cross-attention Block
# -------------------------

class ConceptCrossAttention(nn.Module):
    """
    Single cross-attention layer:
      Queries  : K learnable concept tokens
      Key/Value: flattened vision tokens
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor, need_weights: bool = False):
        """
        queries: [B, K, d]
        kv:      [B, N, d]
        returns:
          out: [B, K, d]
          attn_weights: [B, n_heads, K, N] if need_weights else None
        """
        # Pre-norm
        q = self.ln1(queries)
        k = self.ln1(kv)
        attn_out, attn_weights = self.mha(q, k, k, need_weights=need_weights, average_attn_weights=False)
        x = queries + attn_out
        y = self.ln2(x)
        y = x + self.ff(y)
        return y, attn_weights if need_weights else None


# -------------------------
# Main Model
# -------------------------

class VisionConceptModel(nn.Module):
    """
    End-to-end model:
      - Vision backbone -> tokens
      - Project to d_model
      - K learnable concept tokens + cross-attention
      - Optional per-concept binary classification heads
      - Text projection for the pre-computed concept snippet embeddings
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.K = len(cfg.concept_names)
        self.K_cls = len(cfg.cls_concept_names) if cfg.cls_concept_names is not None else self.K

        # Vision backbone + projection to d_model
        if cfg.vision_backbone == "resnet50":
            self.backbone = ResNetBackbone(
                pretrained=cfg.pretrained_vision if cfg.vision_weights_path is None else False,
                weights_path=cfg.vision_weights_path,
            )
            in_ch = self.backbone.out_dim
            self.to_d = nn.Conv2d(in_ch, cfg.d_model, kernel_size=1) if in_ch != cfg.d_model else nn.Identity()
            self.is_vit = False

        elif cfg.vision_backbone == "dinov2":  
            self.backbone = ViTBackbone(
                repo_path=cfg.dinov2_repo_path,
                weights_path=cfg.dinov2_weights_path
            )
            in_ch = self.backbone.hidden_dim
            self.to_d = nn.Linear(in_ch, cfg.d_model) if in_ch != cfg.d_model else nn.Identity()
            self.is_vit = True
        elif cfg.vision_backbone == "rad_dino_maira_2":
            if cfg.rad_dino_model_dir is None:
                raise ValueError("cfg.rad_dino_model_dir must be set for rad_dino_maira_2")

            self.backbone = HFDinov2Backbone(model_dir=cfg.rad_dino_model_dir)
            in_ch = self.backbone.hidden_dim
            self.to_d = nn.Linear(in_ch, cfg.d_model) if in_ch != cfg.d_model else nn.Identity()
            self.is_vit = True

        else:
            raise ValueError("Unsupported vision_backbone")

        # Concept tokens
        self.concept_tokens = nn.Parameter(
            torch.randn(self.K, cfg.d_model) * cfg.concept_token_init_scale
        )

        # Cross-attention
        self.xattn = ConceptCrossAttention(cfg.d_model, cfg.n_heads, cfg.dropout)

        # Per-concept binary classification heads (logit per concept)
        self.cls_heads = None
        self._cls_map_is_identity = False
        self.cls_head_map = cfg.cls_head_map
        if cfg.use_cls_heads:
            if self.cls_head_map is None:
                if self.K_cls != self.K:
                    raise ValueError(
                        "cls_head_map required when cls_concept_names length differs from concept_names."
                    )
                self.cls_head_map = list(range(self.K_cls))

            if len(self.cls_head_map) != self.K_cls:
                raise ValueError("cls_head_map length must match cls_concept_names length.")
            if max(self.cls_head_map) >= self.K:
                raise ValueError("cls_head_map index out of range for concept tokens.")

            self._cls_map_is_identity = (
                self.K_cls == self.K and self.cls_head_map == list(range(self.K_cls))
            )
            if not self._cls_map_is_identity:
                self.register_buffer(
                    "cls_head_map_idx",
                    torch.tensor(self.cls_head_map, dtype=torch.long),
                    persistent=False,
                )
            else:
                self.cls_head_map_idx = None

            self.cls_heads = nn.ModuleList([nn.Linear(cfg.d_model, 1) for _ in range(self.K_cls)])

        # Text projection for the concept snippet embeddings
        self.text_proj = nn.Linear(cfg.text_in_dim, cfg.d_model) if cfg.project_text else None
        # Optional dropout on projected text features to regularize alignment
        self.text_proj_dropout = nn.Dropout(cfg.dropout) if (cfg.project_text and cfg.dropout > 0.0) else nn.Identity()

    def _vision_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int,int]]]:
        """
        Returns token sequence [B,N,d] and optional (H',W') for heatmaps.
        """
        if not self.is_vit:
            fmap, gh, gw = self.backbone(images)         # [B,C,H',W']
            fmap = self.to_d(fmap)                        # [B,d,H',W']
            tokens = fmap.flatten(2).transpose(1, 2)      # [B,N,d]
            return tokens, (gh, gw)
        else:
            tokens, cls, gh, gw = self.backbone(images)   # [B,N,D], [B,D]
            tokens = _maybe_project(tokens, self.to_d)    # [B,N,d]
            return tokens, (gh, gw)

    def forward(
        self,
        images: torch.Tensor,
        concept_text_emb: Optional[torch.Tensor] = None,    # [B,K,text_in_dim]
        concept_text_mask: Optional[torch.Tensor] = None,   # [B,K] bool; True where present
        need_attn: Optional[bool] = True
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          - v_concepts: [B,K,d]           concept visual embeddings
          - concept_logits: [B,K_cls]     per-concept logits (binary, if enabled)
          - concept_attn: [B,K,H',W']     avg-head attention maps (if available)
          - u_text: [B,K,d]               projected text embeddings (if provided)
        """
        B = images.size(0)
        device = images.device

        # Vision tokens
        vtokens, grid = self._vision_tokens(images)      # [B,N,d], (gh,gw)
        N = vtokens.size(1)

        # Prepare queries
        Z = self.concept_tokens.unsqueeze(0).expand(B, -1, -1)  # [B,K,d]

        # Cross-attention
        v_concepts, attn = self.xattn(Z, vtokens, need_weights=need_attn and self.cfg.record_attn_maps)

        # Classification logits
        logits = None
        if self.cls_heads is not None:
            if self._cls_map_is_identity:
                v_for_cls = v_concepts
            else:
                v_for_cls = v_concepts.index_select(1, self.cls_head_map_idx)
            logits = torch.stack(
                [head(v_for_cls[:, i, :]).squeeze(-1) for i, head in enumerate(self.cls_heads)],
                dim=1,
            )  # [B,K_cls]

        # Attention maps
        concept_attn = None
        if attn is not None and grid is not None:
            # attn: [B, n_heads, K, N] -> avg over heads -> [B,K,N] -> reshape to H',W'
            attn_avg = attn.mean(dim=1)            # [B,K,N]
            gh, gw = grid
            concept_attn = attn_avg.reshape(B, self.K, gh, gw)  # not upsampled here

            # Text projection for snippets
        u_text = None
        if concept_text_emb is not None:
            u_text = _maybe_project(concept_text_emb, self.text_proj)  # [B,K,d] or [B,K,768] if no proj
            if self.text_proj is not None:
                u_text = self.text_proj_dropout(u_text)

        return {
            "v_concepts": v_concepts,        # [B,K,d]
            "concept_logits": logits,        # [B,K_cls] or None
            "concept_attn": concept_attn,    # [B,K,H',W'] or None
            "u_text": u_text,                # [B,K,d] or None
        }
