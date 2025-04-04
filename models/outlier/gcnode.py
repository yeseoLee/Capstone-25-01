import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ..base_model import BaseModel


class GCNODE(BaseModel):
    """
    GCN 기반 이상치 탐지 및 보정 모델 (GCN Outlier Detection and Estimation)
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=None,
        num_layers=3,
        dropout=0.2,
    ):
        """
        Args:
            in_channels: 입력 특성 차원
            hidden_channels: 은닉층 차원
            out_channels: 출력 특성 차원 (None이면 in_channels와 동일)
            num_layers: GCN 레이어 수
            dropout: 드롭아웃 비율
        """
        super(GCNODE, self).__init__(name="GCNODE")

        if out_channels is None:
            out_channels = in_channels

        # 네트워크 레이어 정의
        self.convs = nn.ModuleList()

        # 첫 번째 레이어
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # 중간 레이어
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 마지막 레이어
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, data):
        """
        순전파
        Args:
            data: PyG 데이터 객체
                - x: 노드 특성 [num_nodes, in_channels]
                - edge_index: 엣지 인덱스 [2, num_edges]
        """
        x, edge_index = data.x, data.edge_index

        # NaN 값을 0으로 대체 (GNN은 NaN을, 처리하지 못함)
        x = torch.nan_to_num(x, nan=0.0)

        # GCN 레이어 통과
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 마지막 레이어
        x = self.convs[-1](x, edge_index)

        return x
