import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from ..base_model import BaseModel


class BIGSAGE(BaseModel):
    """
    양방향 GraphSAGE 결측치 보간 모델 (Bidirectional GraphSAGE)
    - 시간과 공간 양방향으로 정보를 수집하는 GraphSAGE 모델
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=None,
        num_layers=3,
        dropout=0.2,
        aggr="mean",
    ):
        """
        Args:
            in_channels: 입력 특성 차원
            hidden_channels: 은닉층 차원
            out_channels: 출력 특성 차원 (None이면 in_channels와 동일)
            num_layers: GraphSAGE 레이어 수
            dropout: 드롭아웃 비율
            aggr: 집계 방법 ('mean', 'max', 'sum')
        """
        super(BIGSAGE, self).__init__(name="BIGSAGE")

        if out_channels is None:
            out_channels = in_channels

        # 공간적 네트워크 (노드 간)
        self.spatial_convs = nn.ModuleList()
        self.spatial_convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 2):
            self.spatial_convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.spatial_convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        # 시간적 네트워크 (1D 컨볼루션 사용)
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )

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

        # NaN 값을 0으로 대체
        x = torch.nan_to_num(x, nan=0.0)

        # 공간적 GNN 레이어 통과
        for i, conv in enumerate(self.spatial_convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 차원 변환 (N, C, T) 형태로 for 시간적 컨볼루션
        batch_size, seq_len = x.shape[0], x.shape[1]
        print(batch_size, seq_len)
        x = x.transpose(0, 1).unsqueeze(0)  # (1, seq_len, batch_size)

        # 시간적 컨볼루션
        x = self.temporal_conv(x)
        x = F.relu(x)

        # 원래 차원으로 되돌리기
        x = x.squeeze(0).transpose(0, 1)  # (batch_size, seq_len)

        # 최종 출력 레이어
        x = self.out_layer(x)

        return x
