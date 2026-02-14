import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt.manifolds.stereographic import PoincareBall

from lib.poincare.layers import BusemannPoincareMLR
from lib.lorentz.layers import LorentzMLR
from lib.lorentz.manifold import CustomLorentz
from lib.models.resnet import (
    resnet18,
    resnet50,
    Lorentz_resnet18,
    Lorentz_resnet50
)

from lib.poincare.hnn_manifold import HyperbolicMLR, HNNPlusPlusMLR
from lib.klein.manifold import KleinManifold, KleinManifoldMLR
from lib.Euclidean.mlr import EuclideanMLR
from lib.pv.manifold import PVManifold
from lib.pv.layers import PVManifoldMLR
from lib.poincare.layers import GaneaPoincareMLR

EUCLIDEAN_RESNET_MODEL = {
    18: resnet18,
    50: resnet50
}

LORENTZ_RESNET_MODEL = {
    18: Lorentz_resnet18,
    50: Lorentz_resnet50
}

RESNET_MODEL = {
    "euclidean" : EUCLIDEAN_RESNET_MODEL,
    "lorentz" : LORENTZ_RESNET_MODEL,
}

EUCLIDEAN_DECODER = {
    'mlr' : nn.Linear
}

LORENTZ_DECODER = {
    'mlr' : LorentzMLR
}

POINCARE_DECODER = {    
    'mlr' : BusemannPoincareMLR
}

class ResNetClassifier(nn.Module):
    """ Classifier based on ResNet encoder.
    """
    def __init__(self, 
            num_layers:int, 
            enc_type:str="lorentz", 
            dec_type:str="lorentz",
            enc_kwargs={},
            dec_kwargs={},
            mlr_type='b'  # Add this
        ):
        super(ResNetClassifier, self).__init__()

        self.enc_type = enc_type
        self.dec_type = dec_type
        self.mlr_type = mlr_type

        self.clip_r = dec_kwargs['clip_r']

        self.encoder = RESNET_MODEL[enc_type][num_layers](remove_linear=True, **enc_kwargs) # enc_type=["euclidean","lorentz"]
        self.enc_manifold = self.encoder.manifold

        self.dec_manifold = None
        dec_kwargs['embed_dim']*=self.encoder.block.expansion
        if dec_type == "euclidean":
            self.decoder = nn.Linear(dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
        elif dec_type == "lorentz":
            self.dec_manifold = CustomLorentz(k=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])
            self.decoder = LorentzMLR(self.dec_manifold, dec_kwargs['embed_dim']+1, dec_kwargs['num_classes'])
        elif dec_type == "poincare":
            self.dec_manifold = PoincareBall(c=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])
            if mlr_type == 'b':
                # (feat_dim, num_outcome, c, ball)
                self.decoder = BusemannPoincareMLR(dec_kwargs['embed_dim'], dec_kwargs['num_classes'], dec_kwargs["k"], self.dec_manifold)
            elif mlr_type in ['hnn']:
                self.decoder = HyperbolicMLR(self.dec_manifold, dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
            elif mlr_type == 'g':
                self.decoder = GaneaPoincareMLR(self.dec_manifold, dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
            elif mlr_type == 'hnn++':
                self.decoder = HNNPlusPlusMLR(self.dec_manifold, dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
            else:
                raise ValueError(f"Unknown mlr_type '{mlr_type}' for poincare")
        elif dec_type == "pv":
            self.dec_manifold = PVManifold(c=dec_kwargs["k"])
            # PVManifoldMLR expects curvature c (float), not a manifold instance
            self.decoder = PVManifoldMLR(dec_kwargs["k"], dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
        elif dec_type == "klein":
            self.dec_manifold = KleinManifold(k=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])
            self.decoder = KleinManifoldMLR(self.dec_manifold, dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
        elif dec_type == "euclidean_custom":
            # Use the custom Euclidean MLR head
            self.dec_manifold = None  # Euclidean space does not require a manifold
            self.decoder = EuclideanMLR(None, dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")
        
    def check_manifold(self, x):
        if self.enc_type == "euclidean":
            if self.dec_type in ["poincare", "klein"]:
                x_norm = torch.norm(x, dim=-1, keepdim=True)
                x = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm.clamp_min(1e-15)) * x
                x = self.dec_manifold.expmap0(x)
            elif self.dec_type == "lorentz":
                x = self.dec_manifold.expmap0(F.pad(x, pad=(1,0), value=0))
            elif self.dec_type == "euclidean":
                pass
        elif self.enc_manifold.k != self.dec_manifold.k:
            x = self.dec_manifold.expmap0(self.enc_manifold.logmap0(x))
        
        return x
    
    def embed(self, x):
        x = self.encoder(x)
        embed = self.check_manifold(x)
        return embed

    def forward(self, x):
        x = self.encoder(x)
        x = self.check_manifold(x)
        x = self.decoder(x)
        return x
        
