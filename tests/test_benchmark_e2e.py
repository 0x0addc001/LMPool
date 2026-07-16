import re

from benchmarks.shared_prefix_benchmark import build_prompts


class IdentityChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]


def _locality_groups(prompts):
    return [
        re.search(r"prefix group (locality-\d{4})", prompt).group(1)
        for prompt in prompts
    ]


def test_locality_workload_builds_balanced_distinct_prefix_groups():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=32,
        prompt_repeat=2,
        workload="locality",
        locality_prefix_groups=8,
        seed=7,
    )

    groups = _locality_groups(prompts)
    assert set(groups) == {f"locality-{group:04d}" for group in range(8)}
    assert all(groups.count(group) == 4 for group in set(groups))
    assert groups != [f"locality-{index % 8:04d}" for index in range(32)]


def test_locality_workload_order_is_seeded():
    kwargs = {
        "num_prompts": 32,
        "prompt_repeat": 1,
        "workload": "locality",
        "locality_prefix_groups": 8,
    }

    first = _locality_groups(build_prompts(IdentityChatTokenizer(), seed=3, **kwargs))
    repeated = _locality_groups(build_prompts(IdentityChatTokenizer(), seed=3, **kwargs))
    different = _locality_groups(build_prompts(IdentityChatTokenizer(), seed=4, **kwargs))

    assert first == repeated
    assert first != different
