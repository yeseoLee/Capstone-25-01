from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.typing import OptTensor
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import MLP

from ..layers import PositionalEncoder, TemporalGraphAdditiveAttention


class SPINModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_nodes: int,
        u_size: Optional[int] = None,
        output_size: Optional[int] = None,
        temporal_self_attention: bool = True,
        reweight: Optional[str] = "softmax",
        n_layers: int = 4,
        eta: int = 3,
        message_layers: int = 1,
        support_stgan_format: bool = True,  # STGAN 데이터 형식 지원 여부
    ):
        super(SPINModel, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention
        self.support_stgan_format = support_stgan_format  # STGAN 형식 지원 플래그

        self.u_enc = PositionalEncoder(in_channels=u_size, out_channels=hidden_size, n_layers=2, n_nodes=n_nodes)

        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)

        self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for L in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            encoder = TemporalGraphAdditiveAttention(
                input_size=hidden_size,
                output_size=hidden_size,
                msg_size=hidden_size,
                msg_layers=message_layers,
                temporal_self_attention=temporal_self_attention,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=L < eta,
                norm=True,
                root_weight=True,
                dropout=0.0,
            )
            readout = MLP(hidden_size, hidden_size, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def _prepare_input(self, x):
        """STGAN 형식의 4D 데이터를 3D 형식으로 변환하거나 그대로 사용"""
        if self.support_stgan_format and len(x.shape) == 4:
            # STGAN 형식: [시간, 노드, 특성, 채널] -> [시간, 노드, 특성*채널]
            batch_size, n_nodes, n_features, n_channels = x.shape
            x = x.reshape(batch_size, n_nodes, n_features * n_channels)
        return x

    def _prepare_mask(self, mask):
        """STGAN 형식의 4D 마스크를 3D 형식으로 변환하거나 그대로 사용"""
        if self.support_stgan_format and len(mask.shape) == 4:
            # STGAN 형식: [시간, 노드, 특성, 채널] -> [시간, 노드, 특성*채널]
            batch_size, n_nodes, n_features, n_channels = mask.shape
            mask = mask.reshape(batch_size, n_nodes, n_features * n_channels)
        return mask

    def forward(
        self,
        x: Tensor,
        u: Tensor,
        mask: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        node_index: OptTensor = None,
        target_nodes: OptTensor = None,
    ):
        if target_nodes is None:
            target_nodes = slice(None)

        # STGAN 형식 지원 처리
        x = self._prepare_input(x)
        mask = self._prepare_mask(mask)

        # Whiten missing values
        x = x * mask

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        imputations = []

        for L in range(self.n_layers):
            if L == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(token_index=node_index)
                masked = self.mask_emb(token_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[L](x) * mask  # skip connection for valid x
            h = self.encoder[L](h, edge_index, mask=mask)
            # Read from H to get imputations
            target_readout = self.readout[L](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)

        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list("--hidden-size", type=int, tunable=True, default=32, options=[32, 64, 128, 256])
        parser.add_argument("--u-size", type=int, default=None)
        parser.add_argument("--output-size", type=int, default=None)
        parser.add_argument("--temporal-self-attention", type=bool, default=True)
        parser.add_argument("--reweight", type=str, default="softmax")
        parser.add_argument("--n-layers", type=int, default=4)
        parser.add_argument("--eta", type=int, default=3)
        parser.add_argument("--message-layers", type=int, default=1)
        parser.add_argument("--support-stgan-format", type=bool, default=True)
        return parser
