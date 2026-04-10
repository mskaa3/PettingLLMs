import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.utils.openai import (
    build_agent_address_mapping,
    clear_trajectory_store,
    create_dummy_model_client,
    get_trajectory_store,
    patch_all,
    start_flow_context,
    wrap_autogen_graph,
)

logger = logging.getLogger(__name__)


class GraphExecutorAgent(Agent):
    """
    Executes a graph workflow inside turn-mode using the graph patching utilities.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.extra_trajectory_per_policy: Dict[str, List] = {}
        self.execution_score = 0.0

        # Runtime context is injected by the turn engine before step().
        self.server_address_dict = {}
        self.tokenizer_dict = {}
        self.ppo_trainer_config_dict = {}
        self.processor_dict = {}
        self.agent_policy_mapping = {}
        self.agent_lora_mapping = {}
        self.agent_config_dict = {}

        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def _extract_final_answer(self, text: str) -> str:
        if not text:
            return ""
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, text)
        if boxed_matches:
            return boxed_matches[-1].strip()

        answer_patterns = [
            r"[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$)",
            r"[Tt]he answer is:?\s*(.+?)(?:\n|$)",
            r"[Aa]nswer:?\s*(.+?)(?:\n|$)",
        ]
        for pattern in answer_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _validate_graph_spec(self, team_spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[List[str]], str, int]:
        if not isinstance(team_spec, dict):
            raise ValueError("team_spec is not a dict")

        nodes = team_spec.get("nodes", [])
        edges = team_spec.get("edges", [])
        entry_node = str(team_spec.get("entry_node", "")).strip()
        max_messages = team_spec.get("max_messages", 15)
        try:
            max_messages = int(max_messages)
        except Exception:
            max_messages = 15
        max_messages = max(6, min(max_messages, 40))

        if not isinstance(nodes, list) or len(nodes) == 0:
            raise ValueError("team_spec.nodes is empty")

        normalized_nodes: List[Dict[str, Any]] = []
        seen_ids = set()
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", f"node_{idx+1}")).strip() or f"node_{idx+1}"
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            subtask_ids = node.get("subtask_ids", [])
            if not isinstance(subtask_ids, list):
                subtask_ids = []
            subtask_ids = [str(s).strip() for s in subtask_ids if str(s).strip()]
            normalized_nodes.append(
                {
                    "id": node_id,
                    "subtask_ids": subtask_ids,
                }
            )
        if len(normalized_nodes) == 0:
            raise ValueError("No valid nodes in team_spec")

        node_ids = {n["id"] for n in normalized_nodes}
        normalized_edges: List[List[str]] = []
        if isinstance(edges, list):
            for edge in edges:
                if not isinstance(edge, list) or len(edge) != 2:
                    continue
                src = str(edge[0]).strip()
                dst = str(edge[1]).strip()
                if src in node_ids and dst in node_ids and src != dst:
                    normalized_edges.append([src, dst])

        if entry_node not in node_ids:
            entry_node = normalized_nodes[0]["id"]

        return normalized_nodes, normalized_edges, entry_node, max_messages

    def _build_system_message(self, subtask_text: str) -> str:
        return (
            "You are a collaborative math agent in a graph workflow. "
            "Use prior context, solve your assigned parts rigorously, and when appropriate provide an explicit final answer in \\boxed{} format.\n"
            f"Focus subtasks:\n{subtask_text}"
        )

    def set_runtime_context(
        self,
        server_address_dict: Dict[str, Any],
        tokenizer_dict: Dict[str, Any],
        ppo_trainer_config_dict: Dict[str, Any],
        processor_dict: Dict[str, Any],
        agent_policy_mapping: Dict[str, str],
        agent_lora_mapping: Dict[str, str],
        agent_config_dict: Dict[str, Any],
    ):
        self.server_address_dict = server_address_dict or {}
        self.tokenizer_dict = tokenizer_dict or {}
        self.ppo_trainer_config_dict = ppo_trainer_config_dict or {}
        self.processor_dict = processor_dict or {}
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.agent_lora_mapping = agent_lora_mapping or {}
        self.agent_config_dict = agent_config_dict or {}

    def update_from_env(self, turn_idx: int, env_data: Env):
        team_spec = getattr(env_data.state, "team_spec", {}) or {}
        decomposition = getattr(env_data.state, "decomposition", {}) or {}

        # This agent should run only on turn 2 (third turn).
        can_run = (turn_idx == 2) and not getattr(env_data, "done", False)
        self.skip_current_turn = not can_run
        if self.skip_current_turn:
            self.current_prompt = {"text": "", "image": None}
            return

        prompt = (
            "You are a graph execution controller.\n"
            "You will execute the designed dynamic multi-agent graph now.\n\n"
            f"Decomposition summary: {decomposition}\n"
            f"Graph spec: {team_spec}\n\n"
            "Acknowledge in one short sentence."
        )
        self.current_prompt = {"text": prompt, "image": None}

    def update_from_model(self, response: str):
        self.current_action = response or ""
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        self.extra_trajectory_per_policy = {}
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
            from autogen_agentchat.ui import Console
            from autogen_agentchat.messages import BaseChatMessage
            from math_verify import parse, verify

            team_spec = getattr(env_data.state, "team_spec", {}) or {}
            decomposition = getattr(env_data.state, "decomposition", {}) or {}
            nodes, edges, entry_node, max_messages = self._validate_graph_spec(team_spec)

            graph_agent_names = [n["id"] for n in nodes]

            available_policies = list(self.server_address_dict.keys())
            if len(available_policies) == 0:
                raise ValueError("GraphExecutorAgent has empty server_address_dict runtime context")
            default_policy = available_policies[0]

            graph_agent_policy_mapping: Dict[str, str] = {}
            for node in nodes:
                agent_name = node["id"]
                graph_agent_policy_mapping[agent_name] = default_policy

            # Build mapping agent -> address for graph execution.
            graph_agent_address_mapping = build_agent_address_mapping(
                agent_names=graph_agent_names,
                agent_policy_mapping=graph_agent_policy_mapping,
                server_address_dict=self.server_address_dict,
            )
            for agent_name in graph_agent_names:
                if agent_name not in graph_agent_address_mapping:
                    graph_agent_address_mapping[agent_name] = self.server_address_dict[graph_agent_policy_mapping[agent_name]]

            # Build dummy clients that will be intercepted by patch_all.
            model_client_dict = {}
            for agent_name in graph_agent_names:
                model_client = create_dummy_model_client("autogen")
                policy_name = graph_agent_policy_mapping[agent_name]
                if hasattr(model_client, "_create_args") and isinstance(model_client._create_args, dict):
                    model_client._create_args["model"] = policy_name
                model_client._agent_name = agent_name
                model_client_dict[agent_name] = model_client

            graph_agent_config_dict = {agent_name: self.agent_config_dict.get(agent_name) for agent_name in graph_agent_names}

            # Install graph patch context for this rollout.
            patch_all(
                server_address_dict=self.server_address_dict,
                tokenizer_dict=self.tokenizer_dict,
                ppo_trainer_config_dict=self.ppo_trainer_config_dict,
                agent_policy_mapping=graph_agent_policy_mapping,
                agent_framework="autogen",
                agent_address_mapping=graph_agent_address_mapping,
                agent_lora_mapping=self.agent_lora_mapping,
                agent_config_dict=graph_agent_config_dict,
                processor_dict=self.processor_dict,
            )

            rollout_idx = int(getattr(self, "rollout_idx", 0) or 0)
            env_idx = int(getattr(self, "env_idx", rollout_idx) or rollout_idx)
            start_flow_context(rollout_idx=rollout_idx, env_idx=env_idx)
            clear_trajectory_store()

            # Build a dynamic graph from team_spec.
            subtasks = decomposition.get("subtasks", []) if isinstance(decomposition, dict) else []
            subtask_by_id = {
                str(s.get("id", "")).strip(): str(s.get("description", "")).strip()
                for s in subtasks
                if isinstance(s, dict)
            }

            node_instances = {}
            builder = DiGraphBuilder()
            for node in nodes:
                node_id = node["id"]
                focus_ids = node.get("subtask_ids", [])
                focus_text = []
                for sid in focus_ids:
                    desc = subtask_by_id.get(sid)
                    if desc:
                        focus_text.append(f"- {sid}: {desc}")
                subtask_text = "\n".join(focus_text) if focus_text else "- Solve the full problem robustly."
                system_message = self._build_system_message(subtask_text=subtask_text)

                agent = AssistantAgent(
                    node_id,
                    model_client=model_client_dict[node_id],
                    system_message=system_message,
                )
                node_instances[node_id] = agent
                builder.add_node(agent)

            builder.set_entry_point(node_instances[entry_node])
            for src, dst in edges:
                builder.add_edge(node_instances[src], node_instances[dst])

            graph = builder.build()
            team = GraphFlow(
                participants=builder.get_participants(),
                graph=graph,
                termination_condition=MaxMessageTermination(max_messages),
            )

            async def _dynamic_graph_runner(env: Env, model_client_dict: Dict[str, Any]):
                task = getattr(env.state, "problem", "")
                try:
                    run_result = await Console(team.run_stream(task=task))
                except Exception:
                    run_result = await team.run(task=task)

                final_solution = ""
                for msg in reversed(getattr(run_result, "messages", []) or []):
                    if isinstance(msg, BaseChatMessage):
                        try:
                            final_solution = msg.to_model_text()
                        except Exception:
                            final_solution = str(msg)
                        break
                    content = getattr(msg, "content", None)
                    if isinstance(content, str) and content.strip():
                        final_solution = content
                        break

                gt = getattr(env.state, "ground_truth_answer", None)
                predicted = self._extract_final_answer(final_solution)
                is_correct = False
                if gt is not None and predicted:
                    try:
                        is_correct = bool(verify(predicted, parse(str(gt))))
                    except Exception:
                        is_correct = False

                final_reward = 1.0 if is_correct else 0.0
                env.final_reward = final_reward
                env.state.final_reward = final_reward
                env.state.reasoning_generated_solution = final_solution
                env.state.reasoning_extracted_answer = predicted
                env.state.reasoning_is_correct = is_correct
                return env

            wrapped_graph = wrap_autogen_graph(_dynamic_graph_runner)
            result_env = await wrapped_graph(env=env_data, model_client_dict=model_client_dict)
            trajectory_store = get_trajectory_store()

            final_reward = float(getattr(result_env, "final_reward", 0.0) or 0.0)
            env_data.state.final_reward = final_reward
            env_data.success = final_reward >= 1.0
            env_data.done = True
            env_data.state.graph_execution_error = None
            env_data.state.graph_execution_result = {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "entry_node": entry_node,
                "team_agents": graph_agent_names,
                "num_hops": len(trajectory_store),
                "final_reward": final_reward,
            }

            per_policy_dataprotos: Dict[str, List] = {}
            for (r_idx, _hop_idx, policy_name), (output_dpr, _response) in trajectory_store.items():
                # Ignore stale entries from other rollouts.
                if r_idx != rollout_idx:
                    continue
                if output_dpr is None or getattr(output_dpr, "batch", None) is None:
                    continue
                output_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward], dtype=np.float32)
                output_dpr.non_tensor_batch["reward"] = np.array([final_reward], dtype=np.float32)
                per_policy_dataprotos.setdefault(policy_name, []).append(output_dpr)

            self.extra_trajectory_per_policy = per_policy_dataprotos
            self.execution_score = final_reward
            self.agent_reward = final_reward
        except Exception as e:
            logger.warning(f"Graph execution failed in GraphExecutorAgent: {e}")
            env_data.state.graph_execution_error = str(e)
            env_data.state.graph_execution_result = {"status": "error", "error": str(e)}
            env_data.state.final_reward = 0.0
            env_data.success = False
            env_data.done = True
            self.extra_trajectory_per_policy = {}
            self.execution_score = 0.0
            self.agent_reward = 0.0

    def calculate_reward(self, env_data: Env):
        self.agent_reward = float(self.execution_score)

    def reset(self):
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.agent_reward = 0.0
        self.execution_score = 0.0
        self.extra_trajectory_per_policy = {}
