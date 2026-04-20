from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from autogen_ext.models.openai import OpenAIChatCompletionClient

from pettingllms.mas_graph.math_graph.math_env import MathEnv
from pettingllms.utils.openai import get_hop_idx, get_rollout_idx


_TREE_RECORD_WRITE_LOCK = threading.Lock()
_TREE_RECORD_ANNOUNCED_PATHS = set()


def extract_answer(text: str) -> str:
    if not text:
        return ""

    boxed_pattern = r"\\boxed\{([^}]+)\}"
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

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines and re.search(r"\d", lines[-1]):
        return lines[-1]
    return ""


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""

    normalized = answer.lower().strip()
    normalized = re.sub(r"[,\$\s]+", "", normalized)
    numeric_match = re.search(r"-?\d+\.?\d*", normalized)
    if numeric_match:
        return numeric_match.group(0)
    return normalized


def check_answer_correctness(generated_answer: str, ground_truth_answer: str) -> bool:
    if not generated_answer or not ground_truth_answer:
        return False

    generated_normalized = normalize_answer(generated_answer)
    ground_truth_normalized = normalize_answer(ground_truth_answer)
    if generated_normalized == ground_truth_normalized:
        return True

    try:
        return abs(float(generated_normalized) - float(ground_truth_normalized)) < 1e-6
    except (TypeError, ValueError):
        return False


def _config_value(config_obj, key: str, default):
    if config_obj is None:
        return default
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    return getattr(config_obj, key, default)


def _safe_filename_fragment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._")
    return normalized or "experiment"


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _extract_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None

    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    for candidate in fenced_matches:
        stripped = candidate.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

    for marker in ("{", "["):
        start = text.find(marker)
        if start == -1:
            continue
        try:
            candidate, _ = json.JSONDecoder().raw_decode(text[start:])
            return json.dumps(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _load_json_response(text: str) -> Optional[Any]:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _coerce_string_list(values) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if str(value).strip()]


def _infer_subtask_kind(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    if any(token in text for token in ["verify", "check", "validate"]):
        return "verification"
    if any(token in text for token in ["compute", "calculate", "arithmetic", "evaluate"]):
        return "calculation"
    if any(token in text for token in ["final", "answer", "synthesis", "conclude"]):
        return "final_synthesis"
    if any(token in text for token in ["plan", "strategy", "reframe", "outline"]):
        return "planning"
    return "reasoning"


def _default_capabilities_for_kind(kind: str) -> List[str]:
    defaults = {
        "verification": ["verification"],
        "calculation": ["calculation", "arithmetic"],
        "final_synthesis": ["final_synthesis", "general_reasoning"],
        "planning": ["planning", "general_reasoning"],
        "reasoning": ["general_reasoning"],
    }
    return defaults.get(kind, ["general_reasoning"])


def _topological_layers(subtasks: List[Dict[str, Any]]) -> Tuple[List[List[str]], List[str], bool]:
    subtask_ids = [subtask["id"] for subtask in subtasks]
    indegree = {subtask_id: 0 for subtask_id in subtask_ids}
    outgoing = defaultdict(list)

    for subtask in subtasks:
        for dependency in subtask["depends_on"]:
            if dependency not in indegree:
                continue
            indegree[subtask["id"]] += 1
            outgoing[dependency].append(subtask["id"])

    layers: List[List[str]] = []
    order: List[str] = []
    queue = deque(sorted([subtask_id for subtask_id, degree in indegree.items() if degree == 0]))

    while queue:
        current_layer = list(queue)
        queue.clear()
        layers.append(current_layer)
        for node in current_layer:
            order.append(node)
            for neighbor in sorted(outgoing.get(node, [])):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

    valid = len(order) == len(subtask_ids)
    return layers, order, valid


def _ensure_terminal_subtask(
    subtasks: List[Dict[str, Any]],
    worker_capabilities: List[str],
    require_final_synthesis_subtask: bool,
) -> Tuple[List[Dict[str, Any]], str]:
    outgoing = defaultdict(list)
    for subtask in subtasks:
        for dependency in subtask["depends_on"]:
            outgoing[dependency].append(subtask["id"])

    leaf_ids = [subtask["id"] for subtask in subtasks if not outgoing.get(subtask["id"])]
    explicit_terminal = next(
        (
            subtask["id"]
            for subtask in subtasks
            if subtask.get("kind") == "final_synthesis" or subtask.get("is_terminal", False)
        ),
        None,
    )
    if explicit_terminal and explicit_terminal in leaf_ids:
        return subtasks, explicit_terminal

    if not require_final_synthesis_subtask and len(leaf_ids) == 1:
        return subtasks, leaf_ids[0]

    terminal_id = "s_final"
    while any(subtask["id"] == terminal_id for subtask in subtasks):
        terminal_id = f"{terminal_id}_x"

    terminal_capability = "final_synthesis" if "final_synthesis" in worker_capabilities else "general_reasoning"
    subtasks.append(
        {
            "id": terminal_id,
            "title": "Synthesize the final answer",
            "description": "Combine the finished subtask results into one coherent solution and produce the final answer.",
            "depends_on": leaf_ids,
            "required_capabilities": [terminal_capability],
            "expected_output": "A concise synthesis ending with Final Answer: <answer>.",
            "success_criteria": "Produces a single final answer grounded in the prior subtasks.",
            "kind": "final_synthesis",
            "is_terminal": True,
        }
    )
    return subtasks, terminal_id


def _fallback_decomposition(
    task: str,
    worker_capabilities: List[str],
    require_final_synthesis_subtask: bool,
) -> Dict[str, Any]:
    primary_capability = "general_reasoning"
    if "planning" in worker_capabilities:
        primary_capability = "planning"

    subtasks = [
        {
            "id": "s1",
            "title": "Plan the solution",
            "description": "Outline a workable strategy for solving the problem.",
            "depends_on": [],
            "required_capabilities": [primary_capability],
            "expected_output": "A short plan that identifies the main mathematical steps.",
            "success_criteria": "Mentions the relevant quantities and a plausible path to the answer.",
            "kind": "planning",
        },
        {
            "id": "s2",
            "title": "Solve the problem",
            "description": f"Solve the problem using the plan and show the computation clearly.\nProblem:\n{task}",
            "depends_on": ["s1"],
            "required_capabilities": ["general_reasoning"],
            "expected_output": "A complete solution ending with Final Answer: <answer>.",
            "success_criteria": "Derives a final answer that can be checked against the ground truth.",
            "kind": "reasoning",
        },
    ]
    subtasks, terminal_id = _ensure_terminal_subtask(subtasks, worker_capabilities, require_final_synthesis_subtask)
    layers, order, _ = _topological_layers(subtasks)
    return {
        "summary": "Fallback two-step plan produced because the decomposer output was missing or invalid.",
        "subtasks": subtasks,
        "subtask_map": {subtask["id"]: subtask for subtask in subtasks},
        "edges": [
            {"source": dependency, "target": subtask["id"]}
            for subtask in subtasks
            for dependency in subtask["depends_on"]
        ],
        "layers": layers,
        "topological_order": order,
        "terminal_subtask_id": terminal_id,
        "valid": False,
        "fallback_used": True,
    }


def _validate_decomposition(
    decomposition: Optional[Dict[str, Any]],
    task: str,
    worker_agents: List[Dict[str, Any]],
    max_subtasks: int,
    require_final_synthesis_subtask: bool,
) -> Dict[str, Any]:
    worker_capabilities = sorted(
        {
            capability
            for agent in worker_agents
            for capability in agent.get("capabilities", [])
        }
    )

    if not isinstance(decomposition, dict) or not isinstance(decomposition.get("subtasks"), list):
        return _fallback_decomposition(task, worker_capabilities, require_final_synthesis_subtask)

    sanitized_subtasks: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, raw_subtask in enumerate(decomposition.get("subtasks", [])[:max_subtasks], start=1):
        if not isinstance(raw_subtask, dict):
            continue
        subtask_id = str(raw_subtask.get("id") or f"s{index}").strip() or f"s{index}"
        while subtask_id in seen_ids:
            subtask_id = f"{subtask_id}_x"
        seen_ids.add(subtask_id)

        title = str(raw_subtask.get("title") or raw_subtask.get("name") or f"Subtask {index}").strip()
        description = str(raw_subtask.get("description") or title).strip()
        kind = str(raw_subtask.get("kind") or _infer_subtask_kind(title, description))
        required_capabilities = _coerce_string_list(raw_subtask.get("required_capabilities"))
        if not required_capabilities:
            required_capabilities = _default_capabilities_for_kind(kind)

        sanitized_subtasks.append(
            {
                "id": subtask_id,
                "title": title,
                "description": description,
                "depends_on": _coerce_string_list(raw_subtask.get("depends_on")),
                "required_capabilities": required_capabilities,
                "expected_output": str(raw_subtask.get("expected_output") or "").strip(),
                "success_criteria": str(raw_subtask.get("success_criteria") or "").strip(),
                "kind": kind,
                "is_terminal": bool(raw_subtask.get("is_terminal", False)),
            }
        )

    if not sanitized_subtasks:
        return _fallback_decomposition(task, worker_capabilities, require_final_synthesis_subtask)

    valid_ids = {subtask["id"] for subtask in sanitized_subtasks}
    for subtask in sanitized_subtasks:
        subtask["depends_on"] = [
            dependency
            for dependency in subtask["depends_on"]
            if dependency in valid_ids and dependency != subtask["id"]
        ]

    sanitized_subtasks, terminal_subtask_id = _ensure_terminal_subtask(
        sanitized_subtasks,
        worker_capabilities,
        require_final_synthesis_subtask,
    )
    layers, order, is_acyclic = _topological_layers(sanitized_subtasks)
    fallback_used = False
    if not is_acyclic:
        fallback_used = True
        for index, subtask in enumerate(sanitized_subtasks):
            subtask["depends_on"] = [] if index == 0 else [sanitized_subtasks[index - 1]["id"]]
        layers, order, _ = _topological_layers(sanitized_subtasks)

    return {
        "summary": str(decomposition.get("summary") or decomposition.get("plan_summary") or "").strip(),
        "subtasks": sanitized_subtasks,
        "subtask_map": {subtask["id"]: subtask for subtask in sanitized_subtasks},
        "edges": [
            {"source": dependency, "target": subtask["id"]}
            for subtask in sanitized_subtasks
            for dependency in subtask["depends_on"]
        ],
        "layers": layers,
        "topological_order": order,
        "terminal_subtask_id": terminal_subtask_id,
        "valid": is_acyclic and not fallback_used,
        "fallback_used": fallback_used,
    }


def _agent_success_rate(memory_snapshot: Dict[str, Any], agent_name: str) -> float:
    stats = memory_snapshot.get("agent_stats", {}).get(agent_name, {})
    attempts = stats.get("attempts", 0)
    if attempts <= 0:
        return 0.5
    return float(stats.get("successes", 0)) / float(attempts)


def _capability_success_rate(memory_snapshot: Dict[str, Any], agent_name: str, capability: str) -> float:
    stats = memory_snapshot.get("agent_stats", {}).get(agent_name, {})
    capability_stats = stats.get("capability_stats", {}).get(capability, {})
    attempts = capability_stats.get("attempts", 0)
    if attempts <= 0:
        return _agent_success_rate(memory_snapshot, agent_name)
    return float(capability_stats.get("successes", 0)) / float(attempts)


def _compatibility_score(
    subtask: Dict[str, Any],
    agent: Dict[str, Any],
    memory_snapshot: Dict[str, Any],
) -> float:
    required_capabilities = set(subtask.get("required_capabilities", []))
    agent_capabilities = set(agent.get("capabilities", []))

    if required_capabilities:
        overlap = len(required_capabilities & agent_capabilities) / float(len(required_capabilities))
        capability_history = sum(
            _capability_success_rate(memory_snapshot, agent["name"], capability)
            for capability in required_capabilities
        ) / float(len(required_capabilities))
    else:
        overlap = 0.6
        capability_history = _agent_success_rate(memory_snapshot, agent["name"])

    text = f"{subtask.get('title', '')} {subtask.get('description', '')} {subtask.get('kind', '')}".lower()
    keyword_hits = sum(1 for capability in agent_capabilities if capability.lower().replace("_", " ") in text)
    keyword_score = min(1.0, keyword_hits / 2.0)
    agent_history = _agent_success_rate(memory_snapshot, agent["name"])

    return _clip(
        0.45 * overlap +
        0.30 * capability_history +
        0.20 * agent_history +
        0.05 * keyword_score
    )


def _choose_best_agent(
    subtask: Dict[str, Any],
    worker_agents: List[Dict[str, Any]],
    memory_snapshot: Dict[str, Any],
) -> Tuple[str, float]:
    scored_agents = [
        (agent["name"], _compatibility_score(subtask, agent, memory_snapshot))
        for agent in worker_agents
    ]
    best_agent, best_score = max(scored_agents, key=lambda item: item[1])
    return best_agent, best_score


def _validate_assignment(
    assignment: Optional[Dict[str, Any]],
    decomposition: Dict[str, Any],
    worker_agents: List[Dict[str, Any]],
    memory_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    subtask_map = decomposition["subtask_map"]
    worker_names = {agent["name"] for agent in worker_agents}
    raw_assignments = assignment.get("assignments", []) if isinstance(assignment, dict) else []
    raw_assignment_map = {}
    for raw_assignment in raw_assignments:
        if not isinstance(raw_assignment, dict):
            continue
        subtask_id = str(raw_assignment.get("subtask_id") or "").strip()
        agent_name = str(raw_assignment.get("agent_name") or "").strip()
        if subtask_id and agent_name:
            raw_assignment_map[subtask_id] = raw_assignment

    assignments = []
    assignments_by_subtask = {}
    for subtask_id in decomposition["topological_order"]:
        subtask = subtask_map[subtask_id]
        raw_assignment = raw_assignment_map.get(subtask_id, {})
        agent_name = str(raw_assignment.get("agent_name") or "").strip()
        if agent_name not in worker_names:
            agent_name, compatibility = _choose_best_agent(subtask, worker_agents, memory_snapshot)
            justification = "Heuristic capability and history match."
            fallback_used = True
        else:
            compatibility = _compatibility_score(
                subtask,
                next(agent for agent in worker_agents if agent["name"] == agent_name),
                memory_snapshot,
            )
            justification = str(raw_assignment.get("justification") or "").strip()
            fallback_used = False

        assignment_item = {
            "subtask_id": subtask_id,
            "agent_name": agent_name,
            "compatibility": compatibility,
            "justification": justification,
            "fallback_used": fallback_used,
        }
        assignments.append(assignment_item)
        assignments_by_subtask[subtask_id] = assignment_item

    return {
        "assignments": assignments,
        "assignments_by_subtask": assignments_by_subtask,
        "execution_order": list(decomposition["topological_order"]),
        "parallel_layers": decomposition["layers"],
    }


def _memory_summary(memory_snapshot: Dict[str, Any], worker_agents: List[Dict[str, Any]]) -> str:
    if not memory_snapshot:
        return "No performance history is available yet."

    lines = []
    for agent in worker_agents:
        stats = memory_snapshot.get("agent_stats", {}).get(agent["name"], {})
        attempts = stats.get("attempts", 0)
        if attempts <= 0:
            lines.append(f"- {agent['name']}: no history yet.")
            continue
        success_rate = stats.get("successes", 0) / float(max(1, attempts))
        avg_partial = stats.get("total_partial_reward", 0.0) / float(max(1, attempts))
        capability_stats = stats.get("capability_stats", {})
        ranked_capabilities = sorted(
            capability_stats.items(),
            key=lambda item: (
                item[1].get("successes", 0) / float(max(1, item[1].get("attempts", 1))),
                item[1].get("attempts", 0),
            ),
            reverse=True,
        )[:2]
        strengths = ", ".join(
            f"{capability}:{bucket.get('successes', 0) / float(max(1, bucket.get('attempts', 1))):.2f}"
            for capability, bucket in ranked_capabilities
        )
        if not strengths:
            strengths = "no capability breakdown yet"
        lines.append(
            f"- {agent['name']}: success={success_rate:.2f}, avg_partial={avg_partial:.2f}, strengths={strengths}"
        )

    decomposer_stats = memory_snapshot.get("decomposer_stats", {})
    if decomposer_stats.get("attempts", 0) > 0:
        lines.append(
            "- decompositions: "
            f"avg_reward={decomposer_stats.get('total_reward', 0.0) / float(max(1, decomposer_stats.get('attempts', 1))):.2f}, "
            f"valid_ratio={decomposer_stats.get('valid_plans', 0) / float(max(1, decomposer_stats.get('attempts', 1))):.2f}"
        )
    return "\n".join(lines) if lines else "No performance history is available yet."


def _tree_record_file_name(env: MathEnv, prefix: str) -> str:
    training_cfg = getattr(env.config, "training", None)
    experiment_name = _config_value(training_cfg, "experiment_name", "experiment")
    date_str = datetime.now().strftime("%Y%m%d")
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    job_suffix = f"_job{_safe_filename_fragment(job_id)}" if job_id else ""
    return f"{prefix}_{_safe_filename_fragment(experiment_name)}_{date_str}{job_suffix}.jsonl"


def _resolve_tree_record_path(env: MathEnv, prefix: str = "decomposition_output") -> str:
    output_dir = os.environ.get("OUTPUT_DIR") or os.environ.get("APPTAINERENV_OUTPUT_DIR")
    if not output_dir:
        output_dir = "/tmp/tmpdir/output" if os.path.isdir("/tmp/tmpdir") else os.path.abspath("output")
    os.makedirs(output_dir, exist_ok=True)
    file_name = _tree_record_file_name(env, prefix)
    return os.path.join(output_dir, file_name)


def _json_safe(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _resolve_tree_record_paths(env: MathEnv, prefix: str = "decomposition_output") -> List[str]:
    primary_path = _resolve_tree_record_path(env, prefix=prefix)
    paths = [primary_path]

    shared_dir = os.environ.get("TREE_RECORD_SHARED_DIR") or os.environ.get("APPTAINERENV_TREE_RECORD_SHARED_DIR")
    if shared_dir:
        os.makedirs(shared_dir, exist_ok=True)
        shared_path = os.path.join(shared_dir, _tree_record_file_name(env, prefix))
        if os.path.abspath(shared_path) not in {os.path.abspath(path) for path in paths}:
            paths.append(shared_path)

    return paths


def _append_tree_record(
    env: MathEnv,
    record: Dict[str, Any],
    prefix: str = "decomposition_output",
) -> List[str]:
    output_paths = _resolve_tree_record_paths(env, prefix=prefix)
    safe_record = _json_safe(record)
    with _TREE_RECORD_WRITE_LOCK:
        for output_path in output_paths:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe_record, ensure_ascii=True) + "\n")
            if output_path not in _TREE_RECORD_ANNOUNCED_PATHS:
                print(f"[decomposer_selector_graph] Writing tree records to {output_path}")
                _TREE_RECORD_ANNOUNCED_PATHS.add(output_path)
    return output_paths


def _serialize_prior_decomposition_candidates(prior_candidates: List[Dict[str, Any]]) -> str:
    payload = []
    for candidate in prior_candidates:
        decomposition = candidate.get("decomposition", {})
        payload.append(
            {
                "decomposition_id": candidate.get("decomposition_id"),
                "summary": decomposition.get("summary", ""),
                "subtasks": [
                    {
                        "id": subtask.get("id"),
                        "title": subtask.get("title"),
                        "kind": subtask.get("kind"),
                    }
                    for subtask in decomposition.get("subtasks", [])
                ],
            }
        )
    return json.dumps(payload, indent=2)


def _serialize_prior_selector_candidates(prior_assignments: List[Dict[str, Any]]) -> str:
    payload = []
    for candidate in prior_assignments:
        payload.append(
            {
                "selection_id": candidate.get("selection_id"),
                "assignments": [
                    {
                        "subtask_id": assignment.get("subtask_id"),
                        "agent_name": assignment.get("agent_name"),
                    }
                    for assignment in candidate.get("assignment_plan", {}).get("assignments", [])
                ],
                "selector_reward": candidate.get("selector_reward"),
            }
        )
    return json.dumps(payload, indent=2)


def _build_decomposer_prompt(
    task: str,
    worker_agents: List[Dict[str, Any]],
    memory_snapshot: Dict[str, Any],
    max_subtasks: int,
    candidate_index: int = 0,
    total_candidates: int = 1,
    prior_candidates: Optional[List[Dict[str, Any]]] = None,
) -> str:
    agent_context = json.dumps(
        [
            {
                "name": agent["name"],
                "description": agent.get("description", ""),
                "capabilities": agent.get("capabilities", []),
            }
            for agent in worker_agents
        ],
        indent=2,
    )
    prior_candidates = prior_candidates or []
    prior_candidate_context = ""
    if prior_candidates:
        prior_candidate_context = (
            "Previously proposed decomposition candidates:\n"
            f"{_serialize_prior_decomposition_candidates(prior_candidates)}\n\n"
        )
    return (
        "You are the Decomposer in a multi-agent math workflow.\n"
        "Break the task into a dependency-aware DAG of subtasks that fits the available worker agents.\n"
        f"Use at most {max_subtasks} subtasks before any auto-added terminal synthesis.\n"
        "Prefer decompositions that expose meaningful parallel branches when useful, but keep them executable.\n"
        f"This is decomposition candidate {candidate_index + 1} of {total_candidates}.\n"
        "When possible, propose a meaningfully different valid decomposition from earlier candidates instead of repeating them.\n"
        "Return JSON only with keys: summary, subtasks.\n"
        "Each subtask must include: id, title, description, depends_on, required_capabilities, expected_output, success_criteria, kind.\n\n"
        f"Task:\n{task}\n\n"
        f"Available worker agents:\n{agent_context}\n\n"
        f"{prior_candidate_context}"
        f"Historical performance summary:\n{_memory_summary(memory_snapshot, worker_agents)}\n"
    )


def _build_selector_prompt(
    task: str,
    decomposition: Dict[str, Any],
    worker_agents: List[Dict[str, Any]],
    memory_snapshot: Dict[str, Any],
    candidate_index: int = 0,
    total_candidates: int = 1,
    prior_assignments: Optional[List[Dict[str, Any]]] = None,
) -> str:
    decomposition_view = {
        "summary": decomposition.get("summary", ""),
        "subtasks": decomposition.get("subtasks", []),
        "edges": decomposition.get("edges", []),
        "layers": decomposition.get("layers", []),
        "terminal_subtask_id": decomposition.get("terminal_subtask_id"),
    }
    agent_context = json.dumps(
        [
            {
                "name": agent["name"],
                "description": agent.get("description", ""),
                "capabilities": agent.get("capabilities", []),
            }
            for agent in worker_agents
        ],
        indent=2,
    )
    prior_assignments = prior_assignments or []
    prior_assignment_context = ""
    if prior_assignments:
        prior_assignment_context = (
            "Previously proposed selector candidates for this decomposition:\n"
            f"{_serialize_prior_selector_candidates(prior_assignments)}\n\n"
        )
    return (
        "You are the Selector in a multi-agent math workflow.\n"
        "Assign one worker agent to each subtask. Optimize for agent-task fit and historical performance.\n"
        f"This is selector candidate {candidate_index + 1} of {total_candidates} for the current decomposition.\n"
        "When possible, produce a different viable assignment from earlier selector candidates.\n"
        "Return JSON only with keys: assignments.\n"
        "Each assignment must include: subtask_id, agent_name, justification.\n\n"
        f"Task:\n{task}\n\n"
        f"Decomposition:\n{json.dumps(decomposition_view, indent=2)}\n\n"
        f"Available worker agents:\n{agent_context}\n\n"
        f"{prior_assignment_context}"
        f"Historical performance summary:\n{_memory_summary(memory_snapshot, worker_agents)}\n"
    )


def _build_worker_prompt(
    task: str,
    subtask: Dict[str, Any],
    assigned_agent: Dict[str, Any],
    dependency_outputs: Dict[str, Dict[str, Any]],
) -> str:
    dependency_context = json.dumps(
        [
            {
                "subtask_id": subtask_id,
                "result": output.get("result", ""),
                "answer_candidate": output.get("answer_candidate", ""),
            }
            for subtask_id, output in dependency_outputs.items()
        ],
        indent=2,
    )
    return (
        f"You are worker agent '{assigned_agent['name']}'.\n"
        f"Profile: {assigned_agent.get('description', '')}\n"
        f"Capabilities: {assigned_agent.get('capabilities', [])}\n"
        "Solve the assigned subtask using the dependency outputs when relevant.\n"
        "Return JSON only with keys: status, result, answer_candidate, confidence.\n"
        "Use status='completed' unless you are clearly blocked.\n\n"
        f"Original task:\n{task}\n\n"
        f"Assigned subtask:\n{json.dumps(subtask, indent=2)}\n\n"
        f"Dependency outputs:\n{dependency_context}\n"
    )


async def _call_agent(
    model_client_dict: Dict[str, OpenAIChatCompletionClient],
    agent_name: str,
    prompt: str,
) -> str:
    client = model_client_dict.get(agent_name)
    if client is None:
        client = next(iter(model_client_dict.values()))
    response = await client.create([{"role": "user", "content": prompt}])
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else str(content)


def _register_next_hop(env: MathEnv, **metadata) -> int:
    hop_idx = get_hop_idx()
    env.state.hop_metadata[hop_idx] = metadata
    return hop_idx


def _parse_worker_output(raw_output: str) -> Dict[str, Any]:
    payload = _load_json_response(raw_output)
    if isinstance(payload, dict):
        result = str(payload.get("result") or "").strip()
        answer_candidate = str(payload.get("answer_candidate") or "").strip()
        if not answer_candidate:
            answer_candidate = extract_answer(result or raw_output)
        return {
            "status": str(payload.get("status") or "completed").strip(),
            "result": result or raw_output.strip(),
            "answer_candidate": answer_candidate,
            "confidence": float(payload.get("confidence", 0.5) or 0.5),
            "raw_output": raw_output,
        }
    return {
        "status": "completed" if raw_output.strip() else "failed",
        "result": raw_output.strip(),
        "answer_candidate": extract_answer(raw_output),
        "confidence": 0.5 if raw_output.strip() else 0.0,
        "raw_output": raw_output,
    }


def _compute_worker_partial_reward(
    subtask: Dict[str, Any],
    assignment: Dict[str, Any],
    execution: Dict[str, Any],
    final_reward: float,
    ground_truth_answer: str,
) -> float:
    reward = 0.0
    if execution.get("status") == "completed":
        reward += 0.25
    if execution.get("result"):
        reward += 0.20
    if len(execution.get("result", "")) > 80:
        reward += 0.10
    reward += 0.20 * float(assignment.get("compatibility", 0.0))
    reward += 0.10 * _clip(execution.get("confidence", 0.0))

    answer_candidate = execution.get("answer_candidate", "")
    if answer_candidate:
        reward += 0.05
        if check_answer_correctness(answer_candidate, ground_truth_answer):
            reward += 0.10

    if subtask.get("kind") == "final_synthesis":
        reward += 0.20 * float(final_reward)
    else:
        reward += 0.10 * float(final_reward)

    return _clip(reward)


def _compute_selector_reward(
    partial_rewards: Dict[str, float],
    assignment_plan: Dict[str, Any],
    final_reward: float,
    orchestration_cfg,
) -> float:
    assignments = assignment_plan.get("assignments", [])
    avg_partial = sum(partial_rewards.values()) / float(max(1, len(partial_rewards)))
    avg_compatibility = sum(assignment.get("compatibility", 0.0) for assignment in assignments) / float(max(1, len(assignments)))
    final_weight = float(_config_value(orchestration_cfg, "selector_final_reward_weight", 0.40))
    partial_weight = float(_config_value(orchestration_cfg, "selector_partial_reward_weight", 0.35))
    compatibility_weight = float(_config_value(orchestration_cfg, "selector_compatibility_weight", 0.25))
    return _clip(
        final_weight * float(final_reward) +
        partial_weight * avg_partial +
        compatibility_weight * avg_compatibility
    )


def _compute_worker_training_reward(partial_reward: float, final_reward: float, orchestration_cfg) -> float:
    partial_weight = float(_config_value(orchestration_cfg, "worker_partial_reward_weight", 0.65))
    final_weight = float(_config_value(orchestration_cfg, "worker_final_reward_weight", 0.35))
    return _clip(partial_weight * partial_reward + final_weight * float(final_reward))


def _compute_decomposer_training_reward(
    heuristic_reward: float,
    avg_selector_reward: float,
    orchestration_cfg,
) -> float:
    heuristic_weight = float(_config_value(orchestration_cfg, "decomposer_heuristic_reward_weight", 0.45))
    selector_weight = float(_config_value(orchestration_cfg, "decomposer_selector_reward_weight", 0.55))
    return _clip(
        heuristic_weight * float(heuristic_reward) +
        selector_weight * float(avg_selector_reward)
    )


def _compute_decomposition_reward(
    decomposition: Dict[str, Any],
    worker_agents: List[Dict[str, Any]],
    memory_snapshot: Dict[str, Any],
) -> float:
    subtasks = decomposition.get("subtasks", [])
    if not subtasks:
        return 0.0

    described_subtasks = sum(
        1
        for subtask in subtasks
        if subtask.get("description") and subtask.get("expected_output") and subtask.get("success_criteria")
    )
    coverage_score = described_subtasks / float(max(1, len(subtasks)))
    branching_score = 1.0 if len(decomposition.get("layers", [])) > 1 else 0.5
    validity_score = 1.0 if decomposition.get("valid", False) else 0.45
    structural_score = _clip(0.45 * coverage_score + 0.25 * branching_score + 0.30 * validity_score)

    feasibility_scores = []
    for subtask in subtasks:
        best_agent, best_score = _choose_best_agent(subtask, worker_agents, memory_snapshot)
        _ = best_agent
        feasibility_scores.append(best_score)
    feasibility_score = sum(feasibility_scores) / float(max(1, len(feasibility_scores)))

    reward = 0.45 * structural_score + 0.55 * feasibility_score
    if decomposition.get("fallback_used", False):
        reward = min(reward, 0.55)
    return _clip(reward)


def _resolve_agent_roles(available_agents: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict[str, Any]]]:
    decomposer = next((agent["name"] for agent in available_agents if agent.get("role") == "decomposer"), None)
    selector = next((agent["name"] for agent in available_agents if agent.get("role") == "selector"), None)

    if decomposer is None and available_agents:
        decomposer = available_agents[0]["name"]
    if selector is None:
        remaining = [agent["name"] for agent in available_agents if agent["name"] != decomposer]
        selector = remaining[0] if remaining else decomposer

    worker_agents = [
        agent
        for agent in available_agents
        if agent["name"] not in {decomposer, selector} and agent.get("role", "worker") == "worker"
    ]
    if not worker_agents:
        worker_agents = [
            agent
            for agent in available_agents
            if agent["name"] not in {decomposer, selector}
        ]
    if not worker_agents:
        worker_agents = list(available_agents)
    return decomposer, selector, worker_agents


def _update_hop_metadata(env: MathEnv, hop_idx: Optional[int], **metadata) -> None:
    if hop_idx is None:
        return
    env.state.hop_metadata.setdefault(hop_idx, {}).update(metadata)


def _branch_sort_key(branch_result: Dict[str, Any]) -> Tuple[float, float, float, float]:
    avg_partial_reward = sum(branch_result.get("partial_rewards", {}).values()) / float(
        max(1, len(branch_result.get("partial_rewards", {})))
    )
    return (
        float(branch_result.get("final_reward", 0.0)),
        float(branch_result.get("selector_reward", 0.0)),
        avg_partial_reward,
        1.0 if branch_result.get("final_answer_candidate") else 0.0,
    )


async def _execute_assignment_branch(
    env: MathEnv,
    model_client_dict: Dict[str, OpenAIChatCompletionClient],
    task: str,
    decomposition: Dict[str, Any],
    assignment_plan: Dict[str, Any],
    selector_hop: int,
    worker_agents: List[Dict[str, Any]],
    orchestration_cfg,
    ground_truth_answer: str,
    tree_group_id: int,
    decomposition_id: str,
    selection_id: str,
) -> Dict[str, Any]:
    worker_map = {agent["name"]: agent for agent in worker_agents}
    subtask_outputs: Dict[str, Dict[str, Any]] = {}
    subtask_hops: Dict[str, int] = {}
    execution_trace: List[Dict[str, Any]] = []
    worker_executions: List[Dict[str, Any]] = []
    branch_id = f"{decomposition_id}:{selection_id}"

    for layer_index, layer in enumerate(decomposition["layers"]):
        for subtask_id in layer:
            subtask = decomposition["subtask_map"][subtask_id]
            assignment = assignment_plan["assignments_by_subtask"][subtask_id]
            worker_name = assignment["agent_name"]
            worker_profile = worker_map.get(worker_name, worker_agents[0])
            dependency_outputs = {
                dependency: subtask_outputs[dependency]
                for dependency in subtask.get("depends_on", [])
                if dependency in subtask_outputs
            }
            worker_prompt = _build_worker_prompt(
                task=task,
                subtask=subtask,
                assigned_agent=worker_profile,
                dependency_outputs=dependency_outputs,
            )
            hop_idx = _register_next_hop(
                env,
                orchestration_role="worker",
                graph_stage="execute",
                logical_agent=worker_name,
                subtask_id=subtask_id,
                subtask_type=subtask.get("kind", "reasoning"),
                tree_group_id=tree_group_id,
                decomposition_id=decomposition_id,
                selection_id=selection_id,
                branch_id=branch_id,
                grpo_uid=f"worker|tree_{tree_group_id}|{decomposition_id}|{selection_id}|{subtask_id}",
            )
            worker_output = await _call_agent(model_client_dict, worker_name, worker_prompt)
            parsed_output = _parse_worker_output(worker_output)
            subtask_hops[subtask_id] = hop_idx
            subtask_outputs[subtask_id] = {
                **parsed_output,
                "subtask_id": subtask_id,
                "agent_name": worker_name,
                "kind": subtask.get("kind", "reasoning"),
                "compatibility": assignment.get("compatibility", 0.0),
            }
            execution_trace.append(
                {
                    "layer_index": layer_index,
                    "subtask_id": subtask_id,
                    "agent_name": worker_name,
                    "hop_idx": hop_idx,
                    "status": parsed_output["status"],
                    "answer_candidate": parsed_output["answer_candidate"],
                }
            )
            worker_executions.append(
                {
                    "subtask_id": subtask_id,
                    "hop_idx": hop_idx,
                    "subtask": subtask,
                    "assignment": assignment,
                    "output": subtask_outputs[subtask_id],
                }
            )

    terminal_subtask_id = decomposition["terminal_subtask_id"]
    terminal_output = subtask_outputs.get(terminal_subtask_id, {})
    final_solution_text = terminal_output.get("raw_output") or terminal_output.get("result", "")
    final_answer_candidate = terminal_output.get("answer_candidate") or extract_answer(final_solution_text)
    final_reward = 1.0 if check_answer_correctness(final_answer_candidate, ground_truth_answer) else 0.0

    partial_rewards: Dict[str, float] = {}
    agent_events: List[Dict[str, Any]] = []
    selector_events: List[Dict[str, Any]] = []
    partial_reward_success_threshold = float(
        _config_value(orchestration_cfg, "partial_reward_success_threshold", 0.55)
    )

    for subtask in decomposition["subtasks"]:
        subtask_id = subtask["id"]
        execution = subtask_outputs.get(subtask_id, {})
        assignment = assignment_plan["assignments_by_subtask"].get(subtask_id, {})
        partial_reward = _compute_worker_partial_reward(
            subtask=subtask,
            assignment=assignment,
            execution=execution,
            final_reward=final_reward,
            ground_truth_answer=ground_truth_answer,
        )
        partial_rewards[subtask_id] = partial_reward

        success = partial_reward >= partial_reward_success_threshold
        worker_training_reward = _compute_worker_training_reward(
            partial_reward=partial_reward,
            final_reward=final_reward,
            orchestration_cfg=orchestration_cfg,
        )
        hop_idx = subtask_hops.get(subtask_id)
        if hop_idx is not None:
            env.state.hop_reward_overrides[hop_idx] = worker_training_reward
            _update_hop_metadata(
                env,
                hop_idx,
                partial_reward=partial_reward,
                training_reward=worker_training_reward,
                branch_final_reward=final_reward,
                branch_answer_candidate=final_answer_candidate,
                branch_success=success,
            )

        agent_events.append(
            {
                "agent_name": assignment.get("agent_name"),
                "required_capabilities": subtask.get("required_capabilities", []),
                "subtask_type": subtask.get("kind", "reasoning"),
                "partial_reward": partial_reward,
                "final_reward": final_reward,
                "success": success,
            }
        )
        selector_events.append(
            {
                "agent_name": assignment.get("agent_name"),
                "subtask_type": subtask.get("kind", "reasoning"),
                "partial_reward": partial_reward,
                "final_reward": final_reward,
                "compatibility": assignment.get("compatibility", 0.0),
                "success": success,
            }
        )

    selector_reward = _compute_selector_reward(
        partial_rewards=partial_rewards,
        assignment_plan=assignment_plan,
        final_reward=final_reward,
        orchestration_cfg=orchestration_cfg,
    )
    env.state.hop_reward_overrides[selector_hop] = selector_reward
    _update_hop_metadata(
        env,
        selector_hop,
        selector_reward=selector_reward,
        branch_final_reward=final_reward,
        branch_answer_candidate=final_answer_candidate,
        average_partial_reward=(
            sum(partial_rewards.values()) / float(max(1, len(partial_rewards)))
            if partial_rewards
            else 0.0
        ),
    )

    return {
        "decomposition_id": decomposition_id,
        "selection_id": selection_id,
        "selector_hop": selector_hop,
        "assignment_plan": assignment_plan,
        "subtask_outputs": subtask_outputs,
        "subtask_hops": subtask_hops,
        "execution_trace": execution_trace,
        "worker_executions": worker_executions,
        "partial_rewards": partial_rewards,
        "final_solution_text": final_solution_text,
        "final_answer_candidate": final_answer_candidate,
        "final_reward": final_reward,
        "selector_reward": selector_reward,
        "agent_events": agent_events,
        "selector_events": selector_events,
    }


async def math_decomposer_selector_graph(
    env: Optional[MathEnv] = None,
    model_client_dict: dict = None,
    model_client: OpenAIChatCompletionClient = None,
):
    if env is None:
        raise ValueError("math_decomposer_selector_graph requires a MathEnv instance.")

    available_agents = list(getattr(env.state, "available_agents", []) or [])
    memory_snapshot = dict(getattr(env.state, "performance_memory_snapshot", {}) or {})
    if not available_agents and model_client is not None:
        available_agents = [{"name": "shared_agent", "role": "worker", "capabilities": ["general_reasoning"], "description": ""}]
        model_client_dict = {"shared_agent": model_client}

    if model_client_dict is None:
        raise ValueError("model_client_dict is required for the decomposer/selector workflow.")

    decomposer_name, selector_name, worker_agents = _resolve_agent_roles(available_agents)
    orchestration_cfg = getattr(env.config, "orchestration", None)
    max_subtasks = int(_config_value(orchestration_cfg, "max_subtasks", 4))
    num_decompositions = max(1, int(_config_value(orchestration_cfg, "num_decompositions", 3)))
    num_selections_per_decomposition = max(
        1,
        int(_config_value(orchestration_cfg, "num_selections_per_decomposition", 3)),
    )
    require_final_synthesis_subtask = bool(
        _config_value(orchestration_cfg, "require_final_synthesis_subtask", True)
    )
    task = env.state.problem or ""
    ground_truth_answer = env.state.ground_truth_answer or ""
    try:
        tree_group_id = int(get_rollout_idx())
    except Exception:
        tree_group_id = 0

    decomposition_candidates: List[Dict[str, Any]] = []
    all_branch_results: List[Dict[str, Any]] = []
    agent_events: List[Dict[str, Any]] = []
    selector_events: List[Dict[str, Any]] = []
    decomposer_events: List[Dict[str, Any]] = []

    for decomposition_index in range(num_decompositions):
        decomposition_id = f"d{decomposition_index}"
        decomposer_prompt = _build_decomposer_prompt(
            task=task,
            worker_agents=worker_agents,
            memory_snapshot=memory_snapshot,
            max_subtasks=max_subtasks,
            candidate_index=decomposition_index,
            total_candidates=num_decompositions,
            prior_candidates=decomposition_candidates,
        )
        decomposer_hop = _register_next_hop(
            env,
            orchestration_role="decomposer",
            graph_stage="decompose",
            logical_agent=decomposer_name,
            tree_group_id=tree_group_id,
            decomposition_id=decomposition_id,
            grpo_uid=f"decomposer|tree_{tree_group_id}",
        )
        decomposer_output = await _call_agent(model_client_dict, decomposer_name, decomposer_prompt)
        decomposition = _validate_decomposition(
            decomposition=_load_json_response(decomposer_output),
            task=task,
            worker_agents=worker_agents,
            max_subtasks=max_subtasks,
            require_final_synthesis_subtask=require_final_synthesis_subtask,
        )
        heuristic_decomposition_reward = _compute_decomposition_reward(
            decomposition=decomposition,
            worker_agents=worker_agents,
            memory_snapshot=memory_snapshot,
        )

        selector_candidate_results: List[Dict[str, Any]] = []
        for selection_index in range(num_selections_per_decomposition):
            selection_id = f"s{selection_index}"
            selector_prompt = _build_selector_prompt(
                task=task,
                decomposition=decomposition,
                worker_agents=worker_agents,
                memory_snapshot=memory_snapshot,
                candidate_index=selection_index,
                total_candidates=num_selections_per_decomposition,
                prior_assignments=selector_candidate_results,
            )
            selector_hop = _register_next_hop(
                env,
                orchestration_role="selector",
                graph_stage="select",
                logical_agent=selector_name,
                tree_group_id=tree_group_id,
                decomposition_id=decomposition_id,
                selection_id=selection_id,
                branch_id=f"{decomposition_id}:{selection_id}",
                grpo_uid=f"selector|tree_{tree_group_id}|{decomposition_id}",
            )
            selector_output = await _call_agent(model_client_dict, selector_name, selector_prompt)
            assignment_plan = _validate_assignment(
                assignment=_load_json_response(selector_output),
                decomposition=decomposition,
                worker_agents=worker_agents,
                memory_snapshot=memory_snapshot,
            )

            branch_result = await _execute_assignment_branch(
                env=env,
                model_client_dict=model_client_dict,
                task=task,
                decomposition=decomposition,
                assignment_plan=assignment_plan,
                selector_hop=selector_hop,
                worker_agents=worker_agents,
                orchestration_cfg=orchestration_cfg,
                ground_truth_answer=ground_truth_answer,
                tree_group_id=tree_group_id,
                decomposition_id=decomposition_id,
                selection_id=selection_id,
            )
            branch_result["selector_prompt"] = selector_prompt
            branch_result["selector_raw_output"] = selector_output
            selector_candidate_results.append(branch_result)
            all_branch_results.append(
                {
                    **branch_result,
                    "decomposition": decomposition,
                    "heuristic_decomposition_reward": heuristic_decomposition_reward,
                }
            )
            agent_events.extend(branch_result["agent_events"])
            selector_events.extend(branch_result["selector_events"])
            selection_trace_record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "record_type": "selection_result",
                "experiment_name": _config_value(getattr(env.config, "training", None), "experiment_name", "experiment"),
                "rollout_tree_id": tree_group_id,
                "task_prompt": task,
                "ground_truth_answer": ground_truth_answer,
                "decomposition_id": decomposition_id,
                "selection_id": selection_id,
                "decomposer_hop": decomposer_hop,
                "selector_hop": branch_result["selector_hop"],
                "decomposition": decomposition,
                "assignment_plan": branch_result["assignment_plan"],
                "worker_executions": branch_result.get("worker_executions", []),
                "execution_trace": branch_result["execution_trace"],
                "partial_rewards": branch_result["partial_rewards"],
                "selector_reward": branch_result["selector_reward"],
                "final_answer_candidate": branch_result["final_answer_candidate"],
                "final_reward": branch_result["final_reward"],
            }
            try:
                env.state.tree_trace_paths = _append_tree_record(
                    env,
                    selection_trace_record,
                    prefix="decomposition_trace",
                )
            except Exception as exc:
                print(f"[decomposer_selector_graph] Failed to append selection trace record: {exc}")

        avg_selector_reward = sum(
            candidate["selector_reward"] for candidate in selector_candidate_results
        ) / float(max(1, len(selector_candidate_results)))
        decomposer_reward = _compute_decomposer_training_reward(
            heuristic_reward=heuristic_decomposition_reward,
            avg_selector_reward=avg_selector_reward,
            orchestration_cfg=orchestration_cfg,
        )
        env.state.hop_reward_overrides[decomposer_hop] = decomposer_reward
        _update_hop_metadata(
            env,
            decomposer_hop,
            heuristic_decomposition_reward=heuristic_decomposition_reward,
            average_selector_reward=avg_selector_reward,
            decomposer_reward=decomposer_reward,
            num_subtasks=len(decomposition.get("subtasks", [])),
            valid_decomposition=decomposition.get("valid", False),
        )

        decomposition_candidates.append(
            {
                "decomposition_id": decomposition_id,
                "decomposer_hop": decomposer_hop,
                "decomposer_prompt": decomposer_prompt,
                "raw_output": decomposer_output,
                "decomposition": decomposition,
                "heuristic_decomposition_reward": heuristic_decomposition_reward,
                "avg_selector_reward": avg_selector_reward,
                "decomposer_reward": decomposer_reward,
                "selector_candidates": selector_candidate_results,
            }
        )
        decomposition_trace_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "record_type": "decomposition_result",
            "experiment_name": _config_value(getattr(env.config, "training", None), "experiment_name", "experiment"),
            "rollout_tree_id": tree_group_id,
            "task_prompt": task,
            "ground_truth_answer": ground_truth_answer,
            "decomposition_id": decomposition_id,
            "decomposer_hop": decomposer_hop,
            "decomposer_prompt": decomposer_prompt,
            "decomposer_raw_output": decomposer_output,
            "decomposition": decomposition,
            "heuristic_decomposition_reward": heuristic_decomposition_reward,
            "avg_selector_reward": avg_selector_reward,
            "decomposer_reward": decomposer_reward,
            "selector_candidates": [
                {
                    "selection_id": branch["selection_id"],
                    "selector_hop": branch["selector_hop"],
                    "assignment_plan": branch["assignment_plan"],
                    "selector_reward": branch["selector_reward"],
                    "final_answer_candidate": branch["final_answer_candidate"],
                    "final_reward": branch["final_reward"],
                }
                for branch in selector_candidate_results
            ],
        }
        try:
            env.state.tree_trace_paths = _append_tree_record(
                env,
                decomposition_trace_record,
                prefix="decomposition_trace",
            )
        except Exception as exc:
            print(f"[decomposer_selector_graph] Failed to append decomposition trace record: {exc}")
        decomposer_events.append(
            {
                "reward": decomposer_reward,
                "num_subtasks": len(decomposition.get("subtasks", [])),
                "valid": decomposition.get("valid", False),
            }
        )

    if not all_branch_results:
        env.state.final_reward = 0.0
        env.final_reward = 0.0
        empty_tree_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "record_type": "empty_tree",
            "experiment_name": _config_value(getattr(env.config, "training", None), "experiment_name", "experiment"),
            "rollout_tree_id": tree_group_id,
            "task_prompt": task,
            "ground_truth_answer": ground_truth_answer,
            "num_decompositions": num_decompositions,
            "num_selections_per_decomposition": num_selections_per_decomposition,
            "reason": "no_branch_results",
        }
        try:
            env.state.tree_trace_paths = _append_tree_record(
                env,
                empty_tree_record,
                prefix="decomposition_trace",
            )
        except Exception as exc:
            print(f"[decomposer_selector_graph] Failed to append empty-tree trace record: {exc}")
        return env

    best_branch = max(all_branch_results, key=_branch_sort_key)
    best_decomposition_entry = next(
        (
            candidate
            for candidate in decomposition_candidates
            if candidate["decomposition_id"] == best_branch["decomposition_id"]
        ),
        decomposition_candidates[0],
    )

    env.state.decomposition_graph = best_decomposition_entry["decomposition"]
    env.state.decomposition_candidates = [
        {
            "decomposition_id": candidate["decomposition_id"],
            "heuristic_decomposition_reward": candidate["heuristic_decomposition_reward"],
            "avg_selector_reward": candidate["avg_selector_reward"],
            "decomposer_reward": candidate["decomposer_reward"],
            "summary": candidate["decomposition"].get("summary", ""),
            "selector_candidates": [
                {
                    "selection_id": branch["selection_id"],
                    "selector_reward": branch["selector_reward"],
                    "final_reward": branch["final_reward"],
                    "final_answer_candidate": branch["final_answer_candidate"],
                    "assignments": branch["assignment_plan"].get("assignments", []),
                }
                for branch in candidate["selector_candidates"]
            ],
        }
        for candidate in decomposition_candidates
    ]
    env.state.assignment_plan = best_branch["assignment_plan"]
    env.state.subtask_results = best_branch["subtask_outputs"]
    env.state.subtask_execution_trace = best_branch["execution_trace"]
    env.state.partial_rewards = best_branch["partial_rewards"]
    env.state.decomposition_reward = best_decomposition_entry["decomposer_reward"]
    env.state.selector_reward = best_branch["selector_reward"]
    env.state.final_answer_candidate = best_branch["final_answer_candidate"]
    env.state.best_branch_summary = {
        "tree_group_id": tree_group_id,
        "decomposition_id": best_branch["decomposition_id"],
        "selection_id": best_branch["selection_id"],
        "heuristic_decomposition_reward": best_decomposition_entry["heuristic_decomposition_reward"],
        "avg_selector_reward": best_decomposition_entry["avg_selector_reward"],
        "decomposer_reward": best_decomposition_entry["decomposer_reward"],
        "selector_reward": best_branch["selector_reward"],
        "final_reward": best_branch["final_reward"],
        "final_answer_candidate": best_branch["final_answer_candidate"],
    }

    env.state.reasoning_generated_solution = best_branch["final_solution_text"]
    env.state.reasoning_generated_solution_history.append(best_branch["final_solution_text"])
    env.state.reasoning_extracted_answer = best_branch["final_answer_candidate"]
    env.state.reasoning_extracted_answer_history.append(best_branch["final_answer_candidate"])
    env.state.reasoning_is_correct = bool(best_branch["final_reward"] > 0.0)

    env.state.performance_memory_update = {
        "agent_events": agent_events,
        "selector_events": selector_events,
        "decomposer_events": decomposer_events,
    }
    env.state.final_reward = max(branch["final_reward"] for branch in all_branch_results)
    env.final_reward = env.state.final_reward

    tree_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": _config_value(getattr(env.config, "training", None), "experiment_name", "experiment"),
        "rollout_tree_id": tree_group_id,
        "task_prompt": task,
        "ground_truth_answer": ground_truth_answer,
        "num_decompositions": num_decompositions,
        "num_selections_per_decomposition": num_selections_per_decomposition,
        "best_branch_summary": env.state.best_branch_summary,
        "final_reward": env.state.final_reward,
        "decomposition_candidates": [
            {
                "decomposition_id": candidate["decomposition_id"],
                "decomposer_hop": candidate["decomposer_hop"],
                "decomposer_prompt": candidate.get("decomposer_prompt", ""),
                "decomposer_raw_output": candidate.get("raw_output", ""),
                "decomposition": candidate["decomposition"],
                "heuristic_decomposition_reward": candidate["heuristic_decomposition_reward"],
                "avg_selector_reward": candidate["avg_selector_reward"],
                "decomposer_reward": candidate["decomposer_reward"],
                "selector_candidates": [
                    {
                        "selection_id": branch["selection_id"],
                        "selector_hop": branch["selector_hop"],
                        "selector_prompt": branch.get("selector_prompt", ""),
                        "selector_raw_output": branch.get("selector_raw_output", ""),
                        "assignment_plan": branch["assignment_plan"],
                        "worker_executions": branch.get("worker_executions", []),
                        "execution_trace": branch["execution_trace"],
                        "partial_rewards": branch["partial_rewards"],
                        "selector_reward": branch["selector_reward"],
                        "final_solution_text": branch["final_solution_text"],
                        "final_answer_candidate": branch["final_answer_candidate"],
                        "final_reward": branch["final_reward"],
                    }
                    for branch in candidate["selector_candidates"]
                ],
            }
            for candidate in decomposition_candidates
        ],
    }
    try:
        env.state.tree_record_paths = _append_tree_record(env, tree_record)
    except Exception as exc:
        print(f"[decomposer_selector_graph] Failed to append decomposition tree record: {exc}")
    final_tree_trace_record = {
        **tree_record,
        "record_type": "final_tree",
    }
    try:
        env.state.tree_trace_paths = _append_tree_record(
            env,
            final_tree_trace_record,
            prefix="decomposition_trace",
        )
    except Exception as exc:
        print(f"[decomposer_selector_graph] Failed to append final tree trace record: {exc}")
    return env
