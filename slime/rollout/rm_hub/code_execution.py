"""Code-execution reward via Gym's code_gen FastAPI service.

slime calls Gym's POST /verify endpoint, which extracts code from sample.response,
runs it against unit tests in sample.metadata["unit_tests"], and returns a scalar reward.

Wire with:
    --custom-rm-path slime.rollout.rm_hub.code_execution.custom_rm
    --code-rm-url    http://<gym-host>:<port>/verify
"""

import asyncio
import logging
import random
import re
import time
import uuid

import aiohttp

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_shared_session: aiohttp.ClientSession | None = None

# Qwen3-think and similar reasoning models emit <think>...</think> as part of their
# natural output. Gym's verify treats any <think> tag in output_text as a reasoning
# format violation and applies a penalty, which would punish correct answers.
# Strip these blocks before sending so only the post-reasoning answer reaches Gym.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).lstrip("\n")


def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(limit=64, enable_cleanup_closed=True)
        timeout = aiohttp.ClientTimeout(total=300)
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


def _build_gym_payload(sample: Sample, unit_tests: dict, model_name: str) -> dict:
    """Construct a CompCodingVerifyRequest payload satisfying Gym's pydantic schema.

    Gym's code_gen verify reads `body.response.output_text`, which on the OpenAI
    Response model is a derived property that aggregates `text` from output[]
    items of type "output_text". So the model response must be embedded as an
    output message, not passed as a top-level output_text field.
    """
    response_text = _strip_think_blocks(sample.response or "")
    if isinstance(sample.prompt, str):
        prompt_text = sample.prompt
    elif isinstance(sample.prompt, list):
        prompt_text = "".join(
            (m.get("content", "") if isinstance(m, dict) else "") for m in sample.prompt
        )
    else:
        prompt_text = ""

    rid = f"slime-{uuid.uuid4().hex}"
    msg_id = f"msg-{uuid.uuid4().hex}"
    return {
        "responses_create_params": {
            "input": prompt_text,
            "model": model_name,
        },
        "response": {
            "id": rid,
            "created_at": time.time(),
            "model": model_name,
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response_text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "temperature": None,
            "top_p": None,
        },
        "verifier_metadata": {"unit_tests": unit_tests},
    }


async def custom_rm(args, sample: Sample, max_retries: int = 5, **kwargs) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    unit_tests = metadata.get("unit_tests")
    if not (sample.response or "").strip() or not unit_tests:
        return 0.0

    url = getattr(args, "code_rm_url", None)
    if not url:
        raise ValueError(
            "--code-rm-url is required when using "
            "slime.rollout.rm_hub.code_execution.custom_rm"
        )
    model_name = (
        getattr(args, "hf_checkpoint", None)
        or getattr(args, "model", None)
        or "slime"
    )

    payload = _build_gym_payload(sample, unit_tests, str(model_name))
    session = _get_shared_session()

    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 422:
                    body = await resp.text()
                    logger.error(f"Gym /verify rejected payload (schema): {body[:500]}")
                    return 0.0
                resp.raise_for_status()
                data = await resp.json()
                reward = data.get("reward")
                if reward is None:
                    logger.warning(f"Gym /verify returned no reward field: {data}")
                    return 0.0
                return float(reward)
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(f"code_rm failed after {attempt + 1} attempts: {e}")
                raise
            backoff = min(2**attempt, 30) + random.random()
            logger.info(
                f"code_rm: {type(e).__name__}, retrying in {backoff:.1f}s "
                f"({attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(backoff)
