import logging
import re
from typing import Any, Dict, List, Tuple

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


def _extract_task_points(text: str) -> List[str]:
    if not text:
        return []

    cleaned = re.sub(r"```(?:text|markdown)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    points: List[str] = []
    bullet_pattern = re.compile(r"^(?:[-*•]|\d+[\.\)])\s+(.*\S)\s*$")
    for line in lines:
        match = bullet_pattern.match(line)
        if match:
            content = match.group(1).strip()
            if content:
                points.append(content)

    if not points:
        # Fallback: split by sentence-like separators.
        chunks = re.split(r"[;\n]+", cleaned)
        for chunk in chunks:
            task = chunk.strip(" -\t\r")
            if task:
                points.append(task)

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for point in points:
        normalized = point.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(point)

    return deduped[:8]


class DecomposerAgent(Agent):
    """First-stage planner that decomposes a problem into subtasks."""

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.validation_score = 0.0
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def _validate_decomposition(self, points: List[str], problem: str) -> Tuple[Dict[str, Any], float]:
        points = [str(p).strip() for p in (points or []) if str(p).strip()]
        points = points[:8]
        subtasks = [{"id": f"S{i+1}", "description": p} for i, p in enumerate(points)]

        # todo: tu bedzie moj wlasny reward
        score = 0.0
        if len(points) >= 2:
            score += 0.7
        elif len(points) == 1:
            score += 0.3
        if problem:
            score += 0.3
        score = min(score, 1.0)

        normalized = {
            "goal": str(problem or ""),
            "task_points": points,
            "subtasks": subtasks,
            "dependencies": [],
        }
        return normalized, float(score)

    def update_from_env(self, turn_idx: int, env_data: Env):
        # This agent should run only on turn 0.
        self.skip_current_turn = turn_idx != 0
        if self.skip_current_turn:
            self.current_prompt = {"text": "", "image": None}
            return

        state = getattr(env_data, "state", None)
        problem = getattr(state, "problem", "")
        previous_decomposition = getattr(state, "decomposition", {}) or {}
        previous_points = previous_decomposition.get("task_points", []) if isinstance(previous_decomposition, dict) else []

        prompt = (
            "You are a task decomposer for multi-agent math solving.\n\n"
            f"Problem:\n{problem}\n\n"
            "Return only a bullet list of 2-6 concise steps (no JSON, no prose intro).\n"
            "Rules:\n"
            "- Each bullet = one executable subtask\n"
            "- Keep order logical from first to last\n"
            "- Prefer short, concrete action phrasing\n"
        )
        if previous_points:
            prompt += (
                "\nPrevious points exist. Improve them if needed:\n"
                + "\n".join([f"- {p}" for p in previous_points])
                + "\n"
            )

        self.current_prompt = {"text": prompt, "image": None}

    def update_from_model(self, response: str):
        self.current_action = response or ""
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        points = _extract_task_points(self.current_action)
        if len(points) == 0:
            logger.warning("DecomposerAgent failed to extract bullet points; using fallback full-problem task")
            problem = getattr(env_data.state, "problem", "") or ""
            points = [f"Solve the full problem: {problem[:120]}".strip()]

        problem = getattr(env_data.state, "problem", "") or ""
        normalized, score = self._validate_decomposition(points, problem)
        env_data.state.decomposition = normalized
        env_data.state.graph_execution_error = None
        self.validation_score = score
        self.agent_reward = score

    def calculate_reward(self, env_data: Env):
        # Keep a stable, decomposition-quality reward.
        self.agent_reward = float(self.validation_score)

    def reset(self):
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.agent_reward = 0.0
        self.validation_score = 0.0
