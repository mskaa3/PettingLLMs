import logging
from typing import Any, Dict, List, Tuple

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


class TeamDesignerAgent(Agent):
    """Second-stage planner that builds a graph spec from decomposition and agent pool."""

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.validation_score = 0.0
        self.agent_pool: List[str] = []
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def _get_task_points(self, decomposition: Dict[str, Any]) -> List[str]:
        if not isinstance(decomposition, dict):
            return []

        task_points = decomposition.get("task_points", [])
        if isinstance(task_points, list):
            points = [str(p).strip() for p in task_points if str(p).strip()]
            if points:
                return points

        subtasks = decomposition.get("subtasks", [])
        points: List[str] = []
        for item in subtasks if isinstance(subtasks, list) else []:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description", "")).strip()
            if desc:
                points.append(desc)
        return points

    def _normalize_agent_pool(self, raw_pool: Any, fallback_n_agents: int = 4) -> List[str]:
        if isinstance(raw_pool, list):
            normalized = [str(x).strip() for x in raw_pool if str(x).strip()]
            if normalized:
                return normalized

        n_agents = max(1, int(fallback_n_agents))
        return [f"agent_{i}" for i in range(1, n_agents + 1)]

    def _build_graph_spec(
        self,
        task_points: List[str],
        agent_pool: List[str],
    ) -> Tuple[Dict[str, Any], float]:
        if len(agent_pool) == 0:
            agent_pool = ["agent_1"]

        # Convert point list into canonical subtask IDs to keep interface stable for executor.
        subtask_ids = [f"S{i+1}" for i in range(len(task_points))]

        assignments: Dict[str, List[str]] = {agent_id: [] for agent_id in agent_pool}
        if len(task_points) > 0:
            # Contiguous split preserving task order across agents.
            for i, sid in enumerate(subtask_ids):
                bucket = min((i * len(agent_pool)) // len(task_points), len(agent_pool) - 1)
                assignments[agent_pool[bucket]].append(sid)

        used_agents = [agent_id for agent_id in agent_pool if len(assignments[agent_id]) > 0]
        if len(used_agents) == 0:
            used_agents = [agent_pool[0]]

        nodes = [{"id": agent_id, "subtask_ids": assignments[agent_id]} for agent_id in used_agents]
        edges = [[used_agents[i], used_agents[i + 1]] for i in range(len(used_agents) - 1)]
        entry_node = used_agents[0]
        max_messages = max(6, min(40, 6 + 2 * len(task_points) + len(edges)))

        # todo: mój własny reward
        score = 0.0
        if len(task_points) >= 2:
            score += 0.4
        elif len(task_points) == 1:
            score += 0.2
        if len(used_agents) >= 1:
            score += 0.2
        if len(edges) >= 1:
            score += 0.2
        if len(task_points) > 0:
            covered = sum(len(assignments[a]) for a in used_agents)
            coverage = covered / max(len(task_points), 1)
            score += 0.2 * coverage
        score = min(1.0, score)

        normalized = {
            "nodes": nodes,
            "edges": edges,
            "entry_node": entry_node,
            "max_messages": max_messages,
        }
        return normalized, float(score)

    def update_from_env(self, turn_idx: int, env_data: Env):
        # This agent should run only on turn 1.
        self.skip_current_turn = turn_idx != 1
        if self.skip_current_turn:
            self.current_prompt = {"text": "", "image": None}
            return

        state = getattr(env_data, "state", None)
        decomposition = getattr(state, "decomposition", {}) or {}
        raw_agent_pool = getattr(state, "agent_pool", [])
        fallback_n_agents = getattr(state, "n_agents", 4)
        self.agent_pool = self._normalize_agent_pool(raw_agent_pool, fallback_n_agents=fallback_n_agents)

        task_points = self._get_task_points(decomposition)
        preview = "\n".join([f"- {p}" for p in task_points[:8]]) if task_points else "- (no task points)"
        prompt = (
            "Deterministic team designer mode is enabled.\n"
            "Graph will be built algorithmically from decomposition and agent pool.\n\n"
            f"Agent pool: {self.agent_pool}\n"
            f"Task points:\n{preview}\n\n"
            "Reply with one short ACK."
        )
        self.current_prompt = {"text": prompt, "image": None}

    def update_from_model(self, response: str):
        self.current_action = response or ""
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        state = getattr(env_data, "state", None)
        decomposition = getattr(state, "decomposition", {}) or {}
        task_points = self._get_task_points(decomposition)

        normalized, score = self._build_graph_spec(task_points=task_points, agent_pool=self.agent_pool)
        env_data.state.team_spec = normalized
        self.validation_score = score
        self.agent_reward = score

    def calculate_reward(self, env_data: Env):
        self.agent_reward = float(self.validation_score)

    def reset(self):
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.agent_reward = 0.0
        self.validation_score = 0.0
        self.agent_pool = []
