from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.nn import inits
from torch_geometric.typing import OptTensor
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import MLP

from ..layers import HierarchicalTemporalGraphAttention, PositionalEncoder


class SPINHierarchicalModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        h_size: int,
        z_size: int,
        n_nodes: int,
        z_heads: int = 1,
        u_size: Optional[int] = None,
        output_size: Optional[int] = None,
        n_layers: int = 5,
        eta: int = 3,
        message_layers: int = 1,
        reweight: Optional[str] = "softmax",
        update_z_cross: bool = True,
        norm: bool = True,
        spatial_aggr: str = "add",
        support_stgan_format: bool = True,  # STGAN 데이터 형식 지원 여부
    ):
        super(SPINHierarchicalModel, self).__init__()

        # 초기화에서 중요 파라미터 저장
        self.input_size = input_size
        u_size = u_size or input_size
        output_size = output_size or input_size
        self.h_size = h_size
        self.z_size = z_size

        self.n_nodes = n_nodes
        self.z_heads = z_heads
        self.n_layers = n_layers
        self.eta = eta
        self.support_stgan_format = support_stgan_format  # STGAN 형식 지원 플래그

        print(f"SPINHierarchicalModel 초기화 - input_size: {input_size}, h_size: {h_size}, z_size: {z_size}")

        self.v = StaticGraphEmbedding(n_nodes, h_size)
        self.lin_v = nn.Linear(h_size, z_size, bias=False)
        self.z = nn.Parameter(torch.Tensor(1, z_heads, n_nodes, z_size))
        inits.uniform(z_size, self.z)
        self.z_norm = LayerNorm(z_size)

        self.u_enc = PositionalEncoder(in_channels=u_size, out_channels=h_size, n_layers=2)

        self.h_enc = MLP(input_size, h_size, n_layers=2)
        self.h_norm = LayerNorm(h_size)

        self.v1 = StaticGraphEmbedding(n_nodes, h_size)
        self.m1 = StaticGraphEmbedding(n_nodes, h_size)

        self.v2 = StaticGraphEmbedding(n_nodes, h_size)
        self.m2 = StaticGraphEmbedding(n_nodes, h_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for L in range(n_layers):
            # input_size를 그대로 사용
            x_skip = nn.Linear(input_size, h_size)
            encoder = HierarchicalTemporalGraphAttention(
                h_size=h_size,
                z_size=z_size,
                msg_size=h_size,
                msg_layers=message_layers,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=L < eta,
                update_z_cross=update_z_cross,
                norm=norm,
                root_weight=True,
                aggr=spatial_aggr,
                dropout=0.0,
            )
            readout = MLP(h_size, z_size, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def _prepare_input(self, x):
        """STGAN 형식의 4D 데이터를 3D 형식으로 변환하거나 그대로 사용"""
        if self.support_stgan_format and len(x.shape) == 4:
            # STGAN 형식: [시간, 노드, 특성, 채널] -> [시간, 노드, 특성*채널]
            batch_size, n_nodes, n_features, n_channels = x.shape
            x = x.reshape(batch_size, n_nodes, n_features * n_channels)
            # 디버깅 정보 출력
            print(f"SPINHierarchicalModel: 입력 데이터 형상 변환 {(batch_size, n_nodes, n_features, n_channels)} -> {x.shape}")
        return x

    def _prepare_mask(self, mask):
        """STGAN 형식의 4D 마스크를 3D 형식으로 변환하거나 그대로 사용"""
        if self.support_stgan_format and len(mask.shape) == 4:
            # STGAN 형식: [시간, 노드, 특성, 채널] -> [시간, 노드, 특성*채널]
            batch_size, n_nodes, n_features, n_channels = mask.shape
            mask = mask.reshape(batch_size, n_nodes, n_features * n_channels)
            # 디버깅 정보 출력
            print(f"SPINHierarchicalModel: 마스크 형상 변환 {(batch_size, n_nodes, n_features, n_channels)} -> {mask.shape}")
        return mask

    def forward(  # noqa: C901
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
        if node_index is None:
            node_index = slice(None)

        # 디버깅: 입력 차원 확인
        print(f"Forward 입력 - x: {x.shape}, u: {u.shape}, mask: {mask.shape}")

        # STGAN 형식 지원 처리
        x_orig_shape = x.shape
        x = self._prepare_input(x)
        mask = self._prepare_mask(mask)

        # 디버깅: 변환 후 차원 확인
        print(f"변환 후 - x: {x.shape}, mask: {mask.shape}")

        # edge_index가 정수형이 아닌 경우 변환
        if edge_index is not None and edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        # POSITIONAL ENCODING #################################################

        # 노드 임베딩 준비
        v_nodes = self.v(token_index=node_index)
        z = self.z[..., node_index, :] + self.lin_v(v_nodes)

        # 텐서 차원 조정
        # 데이터 형상이 [배치, 시간, 노드, 특성] -> [배치, 시간, 노드*특성]으로 변환되었으므로
        # 노드별 처리를 위해 다시 구조화

        # x는 [배치, 시간, 노드*특성]
        batch_size, seq_len = x.shape[0], x.shape[1]
        n_nodes = self.n_nodes if x_orig_shape[2] == self.n_nodes else x_orig_shape[2]

        print(f"배치 크기: {batch_size}, 시퀀스 길이: {seq_len}, 노드 수: {n_nodes}")

        # [배치, 시간, 노드*특성] -> [배치, 시간, 노드, 특성]로 재구성
        # 특성 수는 원래 특성*채널 값을 노드 수로 나눔
        features_per_node = x.shape[2] // n_nodes

        print(f"노드당 특성 수: {features_per_node}")

        # 재구성된 x와 mask
        x_reshaped = x.reshape(batch_size, seq_len, n_nodes, features_per_node)
        mask_reshaped = mask.reshape(batch_size, seq_len, n_nodes, features_per_node)

        print(f"재구성된 x 형상: {x_reshaped.shape}")
        print(f"재구성된 mask 형상: {mask_reshaped.shape}")

        # 이제 노드 차원을 처리하기 위해 텐서 재배열
        # [배치, 시간, 노드, 특성] -> [배치*시간, 노드, 특성]
        x_flat = x_reshaped.reshape(-1, n_nodes, features_per_node)
        mask_flat = mask_reshaped.reshape(-1, n_nodes, features_per_node)

        print(f"평탄화된 x 형상: {x_flat.shape}")
        print(f"평탄화된 mask 형상: {mask_flat.shape}")

        # 시간 임베딩 준비
        q = self.u_enc(u, node_index=node_index, node_emb=v_nodes)

        print(f"q shape: {q.shape}")

        # q를 x_flat과 호환되는 형태로 변환
        if len(q.shape) == 4:  # [배치, 시간, 노드, 특성] 형태
            q_flat = q.reshape(-1, q.shape[2], q.shape[3])
            print(f"q_flat shape: {q_flat.shape}")
        else:
            # 다른 형태의 q 처리
            print(f"예상치 못한 q 형태: {q.shape}")
            q_flat = q.unsqueeze(1).repeat(1, n_nodes, 1)
            print(f"조정된 q_flat shape: {q_flat.shape}")

        # 이제 노드별로 처리
        # h_enc는 [배치*시간, 노드, 특성] -> [배치*시간, 노드, h_size]
        h_flat = self.h_enc(x_flat) + q_flat

        print(f"h_flat shape: {h_flat.shape}")

        # 다시 원래 형태로 복원
        h = h_flat.reshape(batch_size, seq_len, n_nodes, self.h_size)

        print(f"h shape: {h.shape}, z shape: {z.shape}")

        # 이제 mask도 같은 형태로 맞춤
        mask = mask_reshaped

        # 마스크 차원 확인 및 조정
        if mask.shape[-1] != h.shape[-1]:
            print(f"마스크 차원 불일치: mask={mask.shape[-1]}, h={h.shape[-1]}")

            # 마스크 차원 확장 (예: [b, t, n, 12] -> [b, t, n, 16])
            mask_expanded = torch.zeros_like(h, dtype=mask.dtype, device=mask.device)
            c_src = mask.shape[-1]
            c_dst = h.shape[-1]

            # 작은 차원까지만 복사
            mask_expanded[..., :c_src] = mask

            # 나머지는 True로 채움 (값이 있다고 가정)
            if c_dst > c_src:
                mask_expanded[..., c_src:] = True

            mask = mask_expanded
            print(f"마스크 조정 후: {mask.shape}")

        # Normalize features
        h, z = self.h_norm(h), self.z_norm(z)

        # ENCODER #############################################################
        # 나머지 부분은 그대로 유지

        # [batch, time, nodes, features]에서 feature 차원 확인 및 맞추기
        feature_dim = h.shape[-1]

        # v1과 m1 임베딩 가져오기 (노드 수만큼)
        v1 = self.v1(token_index=node_index)  # [n_nodes, h_size]
        m1 = self.m1(token_index=node_index)  # [n_nodes, h_size]

        # 차원 정보 로깅
        print(f"v1 임베딩 크기: {v1.shape}")
        print(f"h 텐서 크기: {h.shape}")
        print(f"q 텐서 크기: {q.shape}")
        print(f"mask 텐서 크기: {mask.shape}")

        # 차원이 맞지 않으면 선형 투영으로 맞추기
        if v1.shape[-1] != feature_dim:
            print(f"임베딩 차원 불일치. v1: {v1.shape[-1]}, h: {feature_dim}")
            projection = torch.nn.functional.linear(v1, torch.eye(v1.shape[-1], feature_dim, device=v1.device))
            v1 = projection

            projection = torch.nn.functional.linear(m1, torch.eye(m1.shape[-1], feature_dim, device=m1.device))
            m1 = projection

        # 임베딩을 h와 q의 형태에 맞게 확장
        # v1, m1: [n_nodes, h_size] -> [1, 1, n_nodes, h_size]
        v1 = v1.unsqueeze(0).unsqueeze(0)
        m1 = m1.unsqueeze(0).unsqueeze(0)

        # 배치와 시간 차원으로 브로드캐스팅하여 처리
        # Replace H in missing entries with queries Q. Then, condition H on two
        # different embeddings to distinguish valid values from masked ones.
        h = torch.where(mask.bool(), h + v1, q + m1)

        imputations = []

        for L in range(self.n_layers):
            if L == self.eta:
                # v2와 m2 임베딩 가져오기
                v2 = self.v2(token_index=node_index)  # [n_nodes, h_size]
                m2 = self.m2(token_index=node_index)  # [n_nodes, h_size]

                # 차원이 맞지 않으면 선형 투영으로 맞추기
                if v2.shape[-1] != feature_dim:
                    print(f"임베딩 차원 불일치. v2: {v2.shape[-1]}, h: {feature_dim}")
                    projection = torch.nn.functional.linear(v2, torch.eye(v2.shape[-1], feature_dim, device=v2.device))
                    v2 = projection

                    projection = torch.nn.functional.linear(m2, torch.eye(m2.shape[-1], feature_dim, device=m2.device))
                    m2 = projection

                # 임베딩을 h의 형태에 맞게 확장
                v2 = v2.unsqueeze(0).unsqueeze(0)
                m2 = m2.unsqueeze(0).unsqueeze(0)

                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                h = torch.where(mask.bool(), h + v2, h + m2)

            # Skip connection from input x
            # x_reshaped와 h의 차원을 맞춤
            x_skip_input = x_reshaped
            try:
                # 차원 확인 로깅
                print(f"x_skip_input shape: {x_skip_input.shape}")
                print(f"h shape: {h.shape}")
                print(f"x_skip 레이어 입력 크기: {self.x_skip[L].in_features}, 출력 크기: {self.x_skip[L].out_features}")

                # 레이어 입력에 맞게 텐서 재배열
                # [배치, 시간, 노드, 특성] -> [배치*시간*노드, 특성]
                bs, ts, ns, fs = x_skip_input.shape
                x_skip_flat = x_skip_input.reshape(-1, fs)

                # 입력 크기가 일치하지 않는 경우 처리
                if fs != self.x_skip[L].in_features:
                    print(f"입력 크기 불일치: 텐서 크기 {fs}, 레이어 입력 크기 {self.x_skip[L].in_features}")
                    # 대체 방법: 선형 레이어를 건너뛰고 투사 수행
                    x_skip_output = torch.nn.functional.linear(x_skip_flat, torch.eye(fs, self.h_size, device=x_skip_flat.device))
                else:
                    # 선형 레이어 통과
                    x_skip_output = self.x_skip[L](x_skip_flat)

                # 원래 차원으로 복원
                x_skip_output = x_skip_output.reshape(bs, ts, ns, self.h_size)

                # skip connection 적용
                h = h + x_skip_output * mask
            except RuntimeError as e:
                print(f"x_skip 연산 오류: {e}")
                # 오류 발생 시 건너뛰기
                print("Skip connection 건너뛰기")

            # Masked Temporal GAT for encoding representation
            h, z = self.encoder[L](h, z, edge_index, mask=mask)
            target_readout = self.readout[L](h[..., target_nodes, :])
            imputations.append(target_readout)

        x_hat = imputations.pop(-1)

        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list("--h-size", type=int, tunable=True, default=32, options=[16, 32])
        parser.opt_list("--z-size", type=int, tunable=True, default=32, options=[32, 64, 128])
        parser.opt_list("--z-heads", type=int, tunable=True, default=2, options=[1, 2, 4, 6])
        parser.add_argument("--u-size", type=int, default=None)
        parser.add_argument("--output-size", type=int, default=None)
        parser.opt_list("--encoder-layers", type=int, tunable=True, default=2, options=[1, 2, 3, 4])
        parser.opt_list("--decoder-layers", type=int, tunable=True, default=2, options=[1, 2, 3, 4])
        parser.add_argument("--message-layers", type=int, default=1)
        parser.opt_list("--reweight", type=str, tunable=True, default="softmax", options=[None, "softmax"])
        parser.add_argument("--update-z-cross", type=bool, default=True)
        parser.opt_list("--norm", type=bool, default=True, tunable=True, options=[True, False])
        parser.opt_list("--spatial-aggr", type=str, tunable=True, default="add", options=["add", "softmax"])
        parser.add_argument("--support-stgan-format", type=bool, default=True)
        return parser
