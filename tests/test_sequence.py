import pickle

from lmpool.engine.sequence import Sequence, SequenceStatus
from lmpool.sampling_parameters import SamplingParams


def test_sequence_block_view_and_round_trip_state():
    params = SamplingParams(temperature=0.7, max_tokens=8, ignore_eos=True, max_model_length=32)
    seq = Sequence([1, 2, 3, 4, 5], block_size=2, sampling_params=params)

    assert seq.num_blocks == 3
    assert seq.last_block_num_tokens == 1
    assert seq.block(0) == [1, 2]
    assert seq.block(1) == [3, 4]
    assert seq.block(2) == [5]
    assert seq.prompt_token_ids == [1, 2, 3, 4, 5]
    assert seq.completion_token_ids == []
    assert seq.remaining_decode_blocks == 4

    seq.append_token(6)
    assert seq.num_tokens == 6
    assert seq.last_token == 6
    assert seq.num_completion_tokens == 1
    assert seq.status == SequenceStatus.WAITING

    seq.is_remote_prefix = True
    seq.remote_gpu_id = 3
    seq.pending_swap_in = [9, 10]
    seq.routed_prefix_hashes = [101, 202]
    seq.prefill_attempts = 2
    seq.preemption_count = 1

    payload = pickle.dumps(seq)
    restored = pickle.loads(payload)

    assert restored.seq_id == seq.seq_id
    assert restored.block_size == 2
    assert restored.token_ids == [6]
    assert restored.num_tokens == 6
    assert restored.num_prompt_tokens == 5
    assert restored.is_remote_prefix is True
    assert restored.remote_gpu_id == 3
    assert restored.pending_swap_in == [9, 10]
    assert restored.routed_prefix_hashes == [101, 202]
    assert restored.prefill_attempts == 2
    assert restored.preemption_count == 1
