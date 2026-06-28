import atexit
import logging
import time
import torch.multiprocessing as mp
from queue import Empty

from lmpool.engine.sequence import Sequence
from lmpool.engine.control_plane import ControlPlaneClient, control_plane_process
from lmpool.engine.data_plane import data_plane_process
from lmpool.sampling_parameters import SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _configure_logging(config: dict):
    level_name = str(config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


class LLMEngine:
    """
    LLM 推理引擎（编排层）

    职责：
    - 作为 launcher 管理所有 rank worker 进程的生命周期
    - 将 prompt 注入 rank 0 worker
    - 聚合各个 worker 的 finished / idle / remote sequence 消息
    """

    def __init__(self, config: dict):
        _configure_logging(config)
        self.config = dict(config)
        self.config.setdefault('use_control_plane_process', True)
        self.world_size = config.get("world_size", 1)
        self._mp_ctx = mp.get_context("spawn")

        self.processes = []
        self.recv_queues = {}   # rank -> Queue，用于接收其他 rank 的消息
        self.send_queues = {}   # rank -> Queue，用于向其他 rank 发送消息
        self.control_plane_request_queue = None
        self.control_plane_response_queues = {}
        self.control_plane_process_handle = None
        self.control_plane_client = None
        self._control_plane_restart_count = 0

        self.remote_finished = set()
        self.remote_inflight_seq_ids = set()

        if self.world_size > 1 and self.config.get('enable_global_pool', False) and self.config.get('use_control_plane_process', True):
            self.control_plane_request_queue = self._mp_ctx.Queue()
            self.control_plane_response_queues = {
                rank: self._mp_ctx.Queue() for rank in range(self.world_size)
            }
            self.control_plane_response_queues[-1] = self._mp_ctx.Queue()

        self._start_control_plane_process()
        if self.control_plane_request_queue is not None:
            self.control_plane_client = ControlPlaneClient(
                -1,
                self.control_plane_request_queue,
                self.control_plane_response_queues[-1],
            )

        for i in range(self.world_size):
            recv_q = self._mp_ctx.Queue()
            send_q = self._mp_ctx.Queue()
            self.recv_queues[i] = recv_q
            self.send_queues[i] = send_q
            process = self._mp_ctx.Process(
                target=data_plane_process,
                args=(
                    self.config,
                    i,
                    send_q,
                    recv_q,
                    self.control_plane_request_queue,
                    self.control_plane_response_queues.get(i),
                ),
            )
            self.processes.append(process)
            process.start()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("model_name_or_path", "gpt2")
        )
        atexit.register(self.exit)

    def _start_control_plane_process(self) -> None:
        if self.control_plane_request_queue is None:
            return
        self.control_plane_process_handle = self._mp_ctx.Process(
            target=control_plane_process,
            args=(self.config, self.control_plane_request_queue, self.control_plane_response_queues),
        )
        self.control_plane_process_handle.start()
        logger.info("control plane process started pid=%s", self.control_plane_process_handle.pid)

    def _ensure_control_plane_process(self) -> None:
        if self.control_plane_request_queue is None:
            return
        if self.control_plane_process_handle is None:
            self._start_control_plane_process()
            return
        if self.control_plane_process_handle.is_alive():
            return
        exitcode = self.control_plane_process_handle.exitcode
        logger.warning(
            "control plane process exited with code %s; restarting",
            exitcode,
        )
        try:
            self.control_plane_process_handle.join(timeout=0)
        except Exception:
            pass
        self._control_plane_restart_count += 1
        self._start_control_plane_process()


    def exit(self):
        for rank, q in self.send_queues.items():
            q.put({"type": "exit"})
        for process in self.processes:
            process.join()
        if self.control_plane_request_queue is not None:
            self.control_plane_request_queue.put({"type": "shutdown"})
        if self.control_plane_process_handle is not None:
            self.control_plane_process_handle.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def _drain_worker_messages(self) -> list[tuple[int, list[int]]]:
        finished = []
        for rank, q in self.recv_queues.items():
            try:
                while True:
                    msg = q.get_nowait()
                    msg_type = msg.get("type")
                    if msg_type == "sequence":
                        target = msg["target"]
                        if target in self.send_queues:
                            self.send_queues[target].put({"type": "sequence", "seq": msg["seq"]})
                            self.remote_inflight_seq_ids.add(msg["seq"].seq_id)
                            self.remote_finished.discard(target)
                        continue
                    if msg_type == "finished":
                        for seq_id, tokens in msg["data"]:
                            finished.append((seq_id, tokens))
                            self.remote_inflight_seq_ids.discard(seq_id)
                    elif msg_type == "idle":
                        self.remote_finished.add(rank)
            except Empty:
                pass
            except Exception as e:
                logger.warning("worker message error from rank %s: %s", rank, e)
                pass
        return finished

    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        """Pump worker messages once and return any newly finished sequences."""
        self._ensure_control_plane_process()
        finished = self._drain_worker_messages()
        if not finished:
            time.sleep(0.01)
        return finished, 0, False


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        seq = Sequence(
            token_ids=self.tokenizer.encode(prompt),
            block_size=self.config['block_size'],
            sampling_params=sampling_params,
        )
        target_rank = 0
        if self.control_plane_client is not None:
            target_rank = self.control_plane_client.route_sequence(seq)
        seq.remote_gpu_id = target_rank
        self.send_queues[target_rank].put({"type": "sequence", "seq": seq})

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """批量推理入口"""
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        expected_num_outputs = len(prompts)
        while len(generated_tokens) < expected_num_outputs:
            try:
                self._ensure_control_plane_process()
                outputs, _, _ = self.step()
            except Exception:
                import traceback
                logger.error("engine pump error")
                traceback.print_exc()
                break
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
