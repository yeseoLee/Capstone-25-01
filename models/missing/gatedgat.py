import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from ..base_model import BaseModel


class GATEDGAT(BaseModel):
    """
    게이트 메커니즘을 가진 GAT 결측치 보간 모델
    - 결측치 주변의 정보를 더 잘 활용하기 위한 게이트 메커니즘 적용
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=None,
        num_layers=2,
        dropout=0.2,
        heads=8,
    ):
        """
        Args:
            in_channels: 입력 특성 차원
            hidden_channels: 은닉층 차원
            out_channels: 출력 특성 차원 (None이면 in_channels와 동일)
            num_layers: GAT 레이어 수
            dropout: 드롭아웃 비율
            heads: 어텐션 헤드 수
        """
        super(GATEDGAT, self).__init__(name="GATEDGAT")

        if out_channels is None:
            out_channels = in_channels

        # GAT 레이어
        self.convs = nn.ModuleList()

        # 첫 번째 레이어
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))

        # 중간 레이어
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))

        # 마지막 GAT 레이어
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False))

        # 게이트 메커니즘
        self.gate = nn.Sequential(nn.Linear(hidden_channels + in_channels, hidden_channels), nn.Sigmoid())

        # 최종 출력 레이어
        self.out_layer = nn.Linear(hidden_channels, out_channels)

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

        # 원본 입력 저장 (게이트 메커니즘에 사용)
        x_orig = x.clone()

        # NaN 값을 0으로 대체
        x = torch.nan_to_num(x, nan=0.0)
        x_orig = torch.nan_to_num(x_orig, nan=0.0)

        # GAT 레이어 통과
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 마지막 GAT 레이어
        x = self.convs[-1](x, edge_index)

        # 게이트 메커니즘
        gate_input = torch.cat([x, x_orig], dim=-1)
        gate_value = self.gate(gate_input)

        # 게이트 적용
        x = gate_value * x + (1 - gate_value) * x_orig

        # 최종 출력 레이어
        x = self.out_layer(x)

        return x
