"""
KV 块跨 GPU 传输模块

提供 transfer out / transfer in 语义的核心原语，基于 NCCL send/recv 实现 GPU 间
KV cache 数据的直接搬运。函数名 `swap_out` / `swap_in` 暂时作为 legacy API 保留。

设计要点：
1. 传输粒度：一个 plan 的所有层、K/V 和 block 打包成一个连续张量
2. 通信协议：已规划路径直接执行单次 send/recv；legacy 路径才交换块索引
3. 支持全局内存池枯竭时的覆盖写入（overwrite）
4. 目标 GPU 空闲块分配由 GlobalBlockManager 协调
"""
import logging
import time
import torch
import torch.distributed as dist
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)
_P2P_GROUPS: dict[tuple[int, int], object] = {}


def _normalize_transfer_pairs(
    pairs: Iterable[tuple[int, int]],
    world_size: int,
) -> list[tuple[int, int]]:
    normalized = set()
    for raw_a, raw_b in pairs:
        a, b = sorted((int(raw_a), int(raw_b)))
        if a == b or a < 0 or b >= world_size:
            continue
        normalized.add((a, b))
    return sorted(normalized)


def _pair_group(first: int, second: int):
    return _P2P_GROUPS.get(tuple(sorted((int(first), int(second)))))


def prewarm_p2p_pairs(
    pairs: Iterable[tuple[int, int]],
    *,
    num_layers: int = 1,
    block_size: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 1,
    num_blocks: int = 1,
) -> list[dict]:
    """Warm each NVLink pair with a representative packed KV payload.

    A one-element marker initializes the communicator but misses allocator,
    kernel, and payload-size effects. Use the same all-layer contiguous payload
    as serving so the observed pair cost does not include obsolete per-layer
    launch overhead. This runs before workers report readiness and is excluded
    from serving throughput and latency.
    """
    if not dist.is_available() or not dist.is_initialized():
        return []
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    normalized_pairs = _normalize_transfer_pairs(pairs, world_size)
    # All world ranks create subgroups in the same order. Pair members run one
    # collective to eagerly initialize the dedicated communicator before P2P.
    for pair in normalized_pairs:
        group = dist.new_group(ranks=list(pair), backend="nccl")
        if rank in pair:
            _P2P_GROUPS[pair] = group
            dist.barrier(group=group, device_ids=[rank])
        dist.barrier(device_ids=[rank])
    payload_shape = (
        max(1, int(num_layers)),
        2,
        max(1, int(num_blocks)),
        max(1, int(block_size)),
        max(1, int(num_kv_heads)),
        max(1, int(head_dim)),
    )
    payload = torch.empty(payload_shape, dtype=torch.float16, device=device)
    observations = []
    for pair_index, (src, dst) in enumerate(normalized_pairs):
        group = _pair_group(src, dst)
        payload.fill_(float(src + 1) if rank == src else 0.0)
        warmup_tag = 70000 + pair_index * 2
        measured_tag = warmup_tag + 1
        # First use creates the lazy peer communicator. It is a startup cost,
        # not the steady-state transfer cost used by admission.
        if rank == src:
            dist.send(payload, dst=dst, tag=warmup_tag, group=group)
        elif rank == dst:
            dist.recv(payload, src=src, tag=warmup_tag, group=group)
        dist.barrier(device_ids=[rank])

        if rank == src:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            dist.send(payload, dst=dst, tag=measured_tag, group=group)
            torch.cuda.synchronize(device)
            observations.append({
                "src_gpu": src,
                "dst_gpu": dst,
                "transfer_bytes": payload.numel() * payload.element_size(),
                "transfer_time_s": time.perf_counter() - started,
            })
        elif rank == dst:
            dist.recv(payload, src=src, tag=measured_tag, group=group)
            if float(payload.flatten()[0].item()) != float(src + 1):
                raise RuntimeError(
                    f"KV transfer P2P prewarm validation failed for pair {(src, dst)}"
                )
        # Every rank enters the same pair sequence, including ranks that do not
        # belong to this pair, so communicator creation cannot race serving.
        dist.barrier(device_ids=[rank])
    return observations

# ============================================================================
# 工具函数
# ============================================================================

def _allocate_empty_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    """
    返回指定 block_id 在 KV cache 中的视图引用（不分配新内存）

    Mini-vLLM 中 KV cache 形状：
    (2, max_cached_blocks, block_size, num_kv_heads, head_dim)
    或每层独立：kv_cache[layer_idx] 形状同上
    具体形状取决于 model_runner 初始化时的分配方式

    参数:
        kv_cache: 单层 KV cache 张量（形状不含 num_layers）
        block_id: 目标块索引
    """
    return kv_cache[:, block_id, ...]


def _compute_tag(block_id: int, layer_idx: int, is_k: bool) -> int:
    """生成 NCCL 通信 tag，避免冲突"""
    # K=0, V=1 -> 编码进 tag
    type_bit = 0 if is_k else 1
    # tag = layer_idx * 2 + type_bit，上限 block_id * 10000 避免冲突
    return block_id * 10000 + layer_idx * 2 + type_bit


def _get_layer_kv(kv_cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize supported KV cache layouts to one layer's (k_cache, v_cache).

    Supported layouts:
    - [(k_cache_layer0, v_cache_layer0), ...]
    - (k_cache, v_cache) for a single layer
    - Tensor shaped (2, num_layers, blocks, block_size, heads, dim)
    - Tensor shaped (2, blocks, block_size, heads, dim) for a single layer
    """
    if isinstance(kv_cache, (list, tuple)):
        if not kv_cache:
            raise RuntimeError("empty kv_cache")
        first = kv_cache[0]
        if isinstance(first, (list, tuple)):
            return kv_cache[layer_idx][0], kv_cache[layer_idx][1]
        if len(kv_cache) == 2 and isinstance(kv_cache[0], torch.Tensor):
            return kv_cache[0], kv_cache[1]

    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.dim() == 6:
            return kv_cache[0, layer_idx], kv_cache[1, layer_idx]
        if kv_cache.dim() == 5:
            return kv_cache[0], kv_cache[1]

    raise RuntimeError(
        "Unsupported kv_cache layout for transfer; expected per-layer "
        "(k_cache, v_cache) pairs or a tensor with leading K/V dimension"
    )


def _pack_layer_blocks(
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
    block_ids: List[int],
) -> torch.Tensor:
    """Pack one layer's K/V blocks into one contiguous P2P payload."""
    if not block_ids:
        return torch.empty(
            (2, 0, *layer_k.shape[1:]),
            dtype=layer_k.dtype,
            device=layer_k.device,
        )
    index = torch.as_tensor(block_ids, dtype=torch.long, device=layer_k.device)
    return torch.stack(
        (layer_k.index_select(0, index), layer_v.index_select(0, index)),
        dim=0,
    ).contiguous()


def _unpack_layer_blocks(
    payload: torch.Tensor,
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
    block_ids: List[int],
) -> None:
    """Scatter one packed layer payload into destination physical blocks."""
    if payload.shape[0] != 2 or payload.shape[1] != len(block_ids):
        raise RuntimeError(
            "packed KV payload does not match destination block count: "
            f"payload={tuple(payload.shape)} blocks={len(block_ids)}"
        )
    if not block_ids:
        return
    index = torch.as_tensor(block_ids, dtype=torch.long, device=layer_k.device)
    layer_k.index_copy_(0, index, payload[0])
    layer_v.index_copy_(0, index, payload[1])


def _empty_layer_payload(
    layer_k: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    return torch.empty(
        (2, num_blocks, *layer_k.shape[1:]),
        dtype=layer_k.dtype,
        device=layer_k.device,
    )


def _pack_all_layer_blocks(
    kv_cache,
    num_layers: int,
    block_ids: List[int],
) -> torch.Tensor:
    """Pack an entire transfer plan into one contiguous NCCL payload."""
    first_k, _ = _get_layer_kv(kv_cache, 0)
    payload = torch.empty(
        (num_layers, 2, len(block_ids), *first_k.shape[1:]),
        dtype=first_k.dtype,
        device=first_k.device,
    )
    if not block_ids:
        return payload
    index = torch.as_tensor(block_ids, dtype=torch.long, device=first_k.device)
    for layer_idx in range(num_layers):
        layer_k, layer_v = _get_layer_kv(kv_cache, layer_idx)
        torch.index_select(layer_k, 0, index, out=payload[layer_idx, 0])
        torch.index_select(layer_v, 0, index, out=payload[layer_idx, 1])
    return payload


def _empty_all_layer_payload(
    kv_cache,
    num_layers: int,
    num_blocks: int,
) -> torch.Tensor:
    first_k, _ = _get_layer_kv(kv_cache, 0)
    return torch.empty(
        (num_layers, 2, num_blocks, *first_k.shape[1:]),
        dtype=first_k.dtype,
        device=first_k.device,
    )


def _unpack_all_layer_blocks(
    payload: torch.Tensor,
    kv_cache,
    block_ids: List[int],
) -> None:
    if payload.shape[0] <= 0 or payload.shape[2] != len(block_ids):
        raise RuntimeError(
            "packed multi-layer KV payload does not match destination blocks: "
            f"payload={tuple(payload.shape)} blocks={len(block_ids)}"
        )
    for layer_idx in range(payload.shape[0]):
        layer_k, layer_v = _get_layer_kv(kv_cache, layer_idx)
        _unpack_layer_blocks(payload[layer_idx], layer_k, layer_v, block_ids)


# ============================================================================
# 协议消息
# ============================================================================

def _send_block_list(blocks: List[int], dst: int):
    """发送块索引列表（用于协商）"""
    tensor = torch.tensor(blocks, dtype=torch.int64, device=f"cuda:{dist.get_rank()}")
    length = torch.tensor([len(blocks)], dtype=torch.int64, device=tensor.device)
    group = _pair_group(dist.get_rank(), dst)
    dist.send(length, dst=dst, tag=99999, group=group)
    if len(blocks) > 0:
        dist.send(tensor, dst=dst, tag=99998, group=group)


def _recv_block_list(src: int) -> List[int]:
    """接收块索引列表"""
    length = torch.tensor([0], dtype=torch.int64, device=f"cuda:{dist.get_rank()}")
    group = _pair_group(dist.get_rank(), src)
    dist.recv(length, src=src, tag=99999, group=group)
    n = length.item()
    if n == 0:
        return []
    tensor = torch.zeros(n, dtype=torch.int64, device=f"cuda:{dist.get_rank()}")
    dist.recv(tensor, src=src, tag=99998, group=group)
    return tensor.tolist()


# ============================================================================
# 传输函数
# ============================================================================

def swap_out(
    local_gpu: int,
    blocks_to_evict: List[int],
    target_gpu: int,
    kv_cache: torch.Tensor,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    target_free_blocks: Optional[List[int]] = None,
) -> List[int]:
    """
    将本地 GPU 上的冷 KV 块搬运到 target_gpu

    参数:
        local_gpu:   源 GPU rank（当前进程）
        blocks_to_evict: 需要换出的本地块索引列表
        target_gpu:  目标 GPU rank
        kv_cache:    单层 KV cache 张量，形状 (2, max_blocks, block_size, num_kv_heads, head_dim)
        num_layers:  模型层数
        block_size:  每块的 token 数
        num_kv_heads: KV head 数
        head_dim:    head 维度
        target_free_blocks: 目标 GPU 上空闲块列表（可选，不传则由对方自行分配）

    协议：
        源端将所有层的 K/V block 打包为一个连续 payload，目标端一次接收后
        scatter 到 prepare 阶段预留的物理块。只有 legacy 调用未提供目标块时
        才先交换 block ID。

    返回:
        target_gpu 上新分配的块索引列表
    """
    logger.info(f"transfer_out: GPU{local_gpu} -> GPU{target_gpu} | blocks={blocks_to_evict}")

    rank = dist.get_rank()
    num_blocks = len(blocks_to_evict)
    device = f"cuda:{rank}"

    # 1. 协商目标块索引
    if rank == local_gpu and rank != target_gpu:
        # 我是发送方：告诉接收方“我要换出这些块”
        _send_block_list(blocks_to_evict, dst=target_gpu)
        if target_free_blocks is not None:
            _send_block_list(target_free_blocks, dst=target_gpu)
        # 接收目标 GPU 分配的空闲块 ID
        target_blocks = _recv_block_list(src=target_gpu)
    elif rank == target_gpu and rank != local_gpu:
        # 我是接收方：接收块列表，分配空闲块
        remote_blocks = _recv_block_list(src=local_gpu)
        # 接收发送方指定的空闲块（如果有，用于覆盖模式）
        specified_blocks = _recv_block_list(src=local_gpu)
        if specified_blocks:
            target_blocks = specified_blocks
        else:
            target_blocks = []  # 由外部预先分配好
        _send_block_list(target_blocks, dst=local_gpu)
    else:
        # 本地传输或单 GPU 场景
        target_blocks = blocks_to_evict  # 同 GPU 内 transfer 直接复用

    if len(target_blocks) != num_blocks:
        raise RuntimeError(
            f"transfer_out negotiated mismatched block counts: "
            f"src={len(blocks_to_evict)} dst={len(target_blocks)}"
        )

    # 2. Pack all layers and blocks into one payload. This removes the
    # per-layer blocking P2P launch sequence from the serving critical path.
    tag = _compute_tag(blocks_to_evict[0], 0, is_k=True) if num_blocks else 0
    group = _pair_group(local_gpu, target_gpu)
    if rank == local_gpu and rank != target_gpu:
        payload = _pack_all_layer_blocks(kv_cache, num_layers, blocks_to_evict)
        dist.send(payload, dst=target_gpu, tag=tag, group=group)
    elif rank == target_gpu and rank != local_gpu:
        payload = _empty_all_layer_payload(kv_cache, num_layers, num_blocks)
        dist.recv(payload, src=local_gpu, tag=tag, group=group)
        _unpack_all_layer_blocks(payload, kv_cache, target_blocks)

    # Blocking send/recv pairs are the synchronization boundary for this
    # point-to-point transfer. Do not use a world-size collective here: in
    # multi-pair runs only the source and target ranks enter this function.
    return target_blocks


def swap_in(
    remote_gpu: int,
    remote_blocks: List[int],
    local_gpu: int,
    kv_cache: torch.Tensor,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    local_target_blocks: Optional[List[int]] = None,
    negotiate_blocks: bool = True,
) -> List[int]:
    """
    从 remote_gpu 拉取 KV 块到本地。

    参数同 transfer out，方向相反。

    协议:
        本地端                              远端
        ─────                              ─────
        send remote_blocks                 recv remote_blocks
        send local_target_blocks (可选)    recv local_target_blocks
        recv local_target_blocks           send local_target_blocks
        逐块 recv K、V                     逐块 send K、V
        barrier                            barrier

    返回:
        本地新分配的块索引列表
    """
    logger.info(f"transfer_in: GPU{remote_gpu} -> GPU{local_gpu} | blocks={remote_blocks}")

    rank = dist.get_rank()
    device = f"cuda:{rank}"

    # Control-plane plans already reserve destination IDs. Skip three metadata
    # round trips on that path; keep negotiation only for the legacy API.
    if not negotiate_blocks:
        if rank == local_gpu and rank != remote_gpu:
            if local_target_blocks is None:
                raise RuntimeError("planned transfer requires local_target_blocks")
            local_blocks = list(local_target_blocks)
        else:
            local_blocks = list(remote_blocks)
    elif rank == local_gpu and rank != remote_gpu:
        _send_block_list(remote_blocks, dst=remote_gpu)
        if local_target_blocks is not None:
            _send_block_list(local_target_blocks, dst=remote_gpu)
        local_blocks = _recv_block_list(src=remote_gpu)
    elif rank == remote_gpu and rank != local_gpu:
        remote_blocks_recv = _recv_block_list(src=local_gpu)
        specified = _recv_block_list(src=local_gpu)
        if specified:
            local_blocks = specified
        else:
            local_blocks = remote_blocks_recv
        _send_block_list(local_blocks, dst=local_gpu)
    else:
        local_blocks = remote_blocks

    if len(local_blocks) != len(remote_blocks):
        raise RuntimeError(
            f"transfer_in negotiated mismatched block counts: "
            f"remote={len(remote_blocks)} local={len(local_blocks)}"
        )

    # 2. One contiguous payload per plan instead of one blocking operation per
    # layer. K/V layout remains explicit in the second dimension.
    tag = _compute_tag(remote_blocks[0], 0, is_k=True) if remote_blocks else 0
    group = _pair_group(remote_gpu, local_gpu)
    if rank == local_gpu and rank != remote_gpu:
        payload = _empty_all_layer_payload(kv_cache, num_layers, len(remote_blocks))
        dist.recv(payload, src=remote_gpu, tag=tag, group=group)
        _unpack_all_layer_blocks(payload, kv_cache, local_blocks)
    elif rank == remote_gpu and rank != local_gpu:
        payload = _pack_all_layer_blocks(kv_cache, num_layers, remote_blocks)
        dist.send(payload, dst=local_gpu, tag=tag, group=group)

    # Blocking send/recv pairs are the synchronization boundary for this
    # point-to-point transfer. A world-size collective would deadlock when only
    # one NVLink pair is transferring inside a larger process group.
    return local_blocks


# ============================================================================
# 块分配协商
# ============================================================================

def request_free_blocks(target_gpu: int, num_blocks: int) -> List[int]:
    """
    向目标 GPU 请求分配 num_blocks 个空闲块。
    返回目标 GPU 上空闲块的物理索引列表。

    注意：这只负责传输请求，实际分配需要目标 GPU 上运行的
    GlobalBlockManager.commit_alloc 来执行。
    """
    rank = dist.get_rank()
    device = f"cuda:{rank}"

    request = torch.tensor([num_blocks], dtype=torch.int64, device=device)
    if rank != target_gpu:
        dist.send(request, dst=target_gpu, tag=88888)
        # 接收分配的块列表
        n_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        dist.recv(n_tensor, src=target_gpu, tag=88887)
        n = n_tensor.item()
        if n == 0:
            return []
        blocks = torch.zeros(n, dtype=torch.int64, device=device)
        dist.recv(blocks, src=target_gpu, tag=88886)
        return blocks.tolist()
    else:
        # 目标 GPU 端逻辑由 GlobalBlockManager 触发，此处不应直接运行
        raise RuntimeError("request_free_blocks should not be called on target GPU directly")


def respond_free_blocks(requester_gpu: int, allocated_blocks: List[int]):
    """
    响应空闲块请求，把分配好的块列表发回给请求方。
    在目标 GPU 上由 GlobalBlockManager 调用。
    """
    rank = dist.get_rank()
    device = f"cuda:{rank}"

    if rank != requester_gpu:
        n = len(allocated_blocks)
        n_tensor = torch.tensor([n], dtype=torch.int64, device=device)
        dist.send(n_tensor, dst=requester_gpu, tag=88887)
        if n > 0:
            blocks_tensor = torch.tensor(allocated_blocks, dtype=torch.int64, device=device)
            dist.send(blocks_tensor, dst=requester_gpu, tag=88886)
