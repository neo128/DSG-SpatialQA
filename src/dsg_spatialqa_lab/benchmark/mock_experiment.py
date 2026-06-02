from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dsg_spatialqa_lab.benchmark.experiment_record import (
    experiment_record,
    save_experiment_record,
)
from dsg_spatialqa_lab.benchmark.experiment_summary import (
    experiment_summary_report,
    save_experiment_summary_report,
)
from dsg_spatialqa_lab.benchmark.manifest import (
    build_benchmark_artifacts,
    save_benchmark_manifest,
)
from dsg_spatialqa_lab.benchmark.qa_generator import load_qa_dataset
from dsg_spatialqa_lab.benchmark.qa_generator import save_qa_dataset
from dsg_spatialqa_lab.memory import DynamicSceneGraph
from dsg_spatialqa_lab.scene_io import (
    graph_from_json,
    graph_to_json,
    load_graph_json,
    save_graph_json,
)
from dsg_spatialqa_lab.tasks import ActiveEQATask

if TYPE_CHECKING:
    from dsg_spatialqa_lab.adapters import AI2ThorAdapterConfig


MOCK_EXPERIMENT_RESULT_SCHEMA_VERSION = (
    "dsg-spatialqa-lab.mock-experiment-result.v1"
)


def run_mock_experiment(
    *,
    output_dir: str | Path,
    dataset_name: str = "mock_experiment",
    max_qa_per_episode: int = 3,
    episode_configs: Sequence["AI2ThorAdapterConfig"] | None = None,
    qa_baseline_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    from dsg_spatialqa_lab.adapters import (
        AI2ThorAdapterConfig,
        build_mock_ai2thor_episode,
    )
    from dsg_spatialqa_lab.agents import ActiveGraphAgent, run_baseline_predictions
    from dsg_spatialqa_lab.episodes import save_episode_sequence
    from dsg_spatialqa_lab.eval import (
        active_task_delta_report,
        active_task_report,
        error_attribution_report,
        graph_eval_report,
        qa_eval_delta_report,
        qa_eval_report,
        save_active_task_delta_report,
        save_active_task_report,
        save_error_attribution_report,
        save_graph_eval_report,
        save_qa_eval_delta_report,
        save_qa_eval_report,
        save_qa_predictions,
    )
    from dsg_spatialqa_lab.predicted import (
        build_predicted_graph_from_episode,
        predicted_graph_report,
        save_predicted_graph_report,
    )
    from dsg_spatialqa_lab.tasks import MockActiveEnvironment, save_active_eqa_tasks
    from dsg_spatialqa_lab.visualization import dashboard_bundle, export_dashboard

    root = Path(output_dir)
    benchmark_dir = root / "benchmark"
    predicted_dir = root / "predicted"
    prediction_dir = root / "predictions"
    report_dir = root / "reports"
    active_dir = root / "active"
    dashboard_dir = root / "dashboard"
    manifest_path = root / "benchmark-manifest.json"
    summary_report_path = root / "experiment-summary.json"
    record_path = root / "experiment-record.json"
    configs = _episode_configs(AI2ThorAdapterConfig, episode_configs)
    use_legacy_single_episode_path = episode_configs is None and len(configs) == 1
    qa_candidate_name = "graph_tool"
    qa_graph_construction_baseline_name = "predicted_graph_tool"
    qa_baselines = _qa_baseline_names(qa_baseline_names)
    qa_agent_names = (qa_candidate_name, *qa_baselines)

    episode_paths: list[Path] = []
    episode_frame_groups: list[Any] = []
    for config in configs:
        episode_path = _episode_path(
            root,
            config.episode_id,
            use_legacy_single_episode_path=use_legacy_single_episode_path,
        )
        frames = tuple(build_mock_ai2thor_episode(config))
        save_episode_sequence(frames, episode_path)
        episode_paths.append(episode_path)
        episode_frame_groups.append(frames)

    initial_manifest = build_benchmark_artifacts(
        dataset_name=dataset_name,
        episode_paths=tuple(episode_paths),
        output_dir=benchmark_dir,
        max_qa_per_episode=max_qa_per_episode,
    )
    artifacts = _artifact_items(initial_manifest, expected_count=len(episode_paths))
    graph_paths = tuple(_required_artifact_path(artifact, "graph_path") for artifact in artifacts)
    qa_paths = tuple(_required_artifact_path(artifact, "qa_path") for artifact in artifacts)
    graphs = tuple(load_graph_json(path) for path in graph_paths)
    qa_case_groups = tuple(tuple(load_qa_dataset(path)) for path in qa_paths)
    qa_cases = tuple(case for group in qa_case_groups for case in group)
    combined_qa_path = benchmark_dir / "qa" / f"{_path_stem(dataset_name)}-qa.jsonl"
    save_qa_dataset(qa_cases, combined_qa_path)

    predicted_dir.mkdir(parents=True, exist_ok=True)
    predicted_graph_paths: list[Path] = []
    predicted_graph_report_paths: list[Path] = []
    predicted_graphs: list[DynamicSceneGraph] = []
    graph_eval_report_paths: list[Path] = []
    for graph, graph_path, episode_path, frames in zip(
        graphs,
        graph_paths,
        episode_paths,
        episode_frame_groups,
        strict=True,
    ):
        episode_id = _episode_id_from_frames(frames)
        artifact_name = _path_stem(episode_id)
        predicted_graph = build_predicted_graph_from_episode(frames)
        predicted_graph_path = predicted_dir / f"{artifact_name}-predicted-graph.json"
        predicted_report_path = report_dir / f"predicted-{artifact_name}-report.json"
        graph_eval_report_path = report_dir / f"graph-eval-{artifact_name}.json"
        save_graph_json(predicted_graph, predicted_graph_path)
        predicted_report = predicted_graph_report(
            input_path=episode_path,
            graph_path=predicted_graph_path,
            graph=predicted_graph,
            frames=frames,
        )
        save_predicted_graph_report(predicted_report, predicted_report_path)
        graph_eval = graph_eval_report(
            graph,
            predicted_graph,
            oracle_path=graph_path,
            predicted_path=predicted_graph_path,
        )
        save_graph_eval_report(graph_eval, graph_eval_report_path)
        predicted_graphs.append(predicted_graph)
        predicted_graph_paths.append(predicted_graph_path)
        predicted_graph_report_paths.append(predicted_report_path)
        graph_eval_report_paths.append(graph_eval_report_path)

    qa_predictions_by_agent: dict[str, Any] = {}
    qa_prediction_paths: dict[str, Path] = {}
    qa_reports_by_agent: dict[str, dict[str, Any]] = {}
    qa_report_paths: dict[str, Path] = {}
    for agent_name in qa_agent_names:
        prediction_path = prediction_dir / f"{_agent_slug(agent_name)}-predictions.jsonl"
        predictions = [
            prediction
            for graph, group in zip(graphs, qa_case_groups, strict=True)
            for prediction in run_baseline_predictions(agent_name, graph=graph, cases=group)
        ]
        save_qa_predictions(predictions, prediction_path)
        report_path = report_dir / f"qa-{_agent_slug(agent_name)}-report.json"
        report = qa_eval_report(
            qa_cases,
            predictions,
            gold_path=combined_qa_path,
            prediction_path=prediction_path,
        )
        save_qa_eval_report(report, report_path)
        qa_predictions_by_agent[agent_name] = predictions
        qa_prediction_paths[agent_name] = prediction_path
        qa_reports_by_agent[agent_name] = report
        qa_report_paths[agent_name] = report_path

    predicted_graph_tool_prediction_path = (
        prediction_dir / "predicted-graph-tool-predictions.jsonl"
    )
    predicted_graph_tool_predictions = [
        prediction
        for graph, group in zip(predicted_graphs, qa_case_groups, strict=True)
        for prediction in run_baseline_predictions("graph_tool", graph=graph, cases=group)
    ]
    save_qa_predictions(
        predicted_graph_tool_predictions,
        predicted_graph_tool_prediction_path,
    )
    predicted_graph_tool_report_path = report_dir / "qa-predicted-graph-tool-report.json"
    predicted_graph_tool_report = qa_eval_report(
        qa_cases,
        predicted_graph_tool_predictions,
        gold_path=combined_qa_path,
        prediction_path=predicted_graph_tool_prediction_path,
    )
    save_qa_eval_report(predicted_graph_tool_report, predicted_graph_tool_report_path)
    qa_predictions_by_agent[qa_graph_construction_baseline_name] = (
        predicted_graph_tool_predictions
    )
    qa_prediction_paths[qa_graph_construction_baseline_name] = (
        predicted_graph_tool_prediction_path
    )
    qa_reports_by_agent[qa_graph_construction_baseline_name] = (
        predicted_graph_tool_report
    )
    qa_report_paths[qa_graph_construction_baseline_name] = (
        predicted_graph_tool_report_path
    )

    error_attribution_report_paths: list[Path] = []
    error_attribution_reports: list[dict[str, Any]] = []
    for graph, predicted_graph, group, qa_path, graph_path, predicted_graph_path, frames in zip(
        graphs,
        predicted_graphs,
        qa_case_groups,
        qa_paths,
        graph_paths,
        predicted_graph_paths,
        episode_frame_groups,
        strict=True,
    ):
        episode_id = _episode_id_from_frames(frames)
        artifact_name = _path_stem(episode_id)
        attribution_report_path = report_dir / f"error-attribution-{artifact_name}.json"
        attribution_report = error_attribution_report(
            group,
            oracle_graph=graph,
            predicted_graph=predicted_graph,
            predictions=predicted_graph_tool_predictions,
            gold_path=qa_path,
            oracle_graph_path=graph_path,
            predicted_graph_path=predicted_graph_path,
            prediction_path=predicted_graph_tool_prediction_path,
        )
        save_error_attribution_report(attribution_report, attribution_report_path)
        error_attribution_reports.append(attribution_report)
        error_attribution_report_paths.append(attribution_report_path)

    qa_candidate_report = qa_reports_by_agent[qa_candidate_name]
    qa_candidate_report_path = qa_report_paths[qa_candidate_name]
    primary_qa_baseline_name = qa_baselines[0]
    qa_baseline_report_path = qa_report_paths[primary_qa_baseline_name]
    graph_tool_predictions = qa_predictions_by_agent[qa_candidate_name]
    graph_tool_prediction_path = qa_prediction_paths[qa_candidate_name]
    majority_prediction_path = qa_prediction_paths.get(
        "majority",
        qa_prediction_paths[primary_qa_baseline_name],
    )

    qa_delta_report_paths: dict[str, Path] = {}
    for index, baseline_name in enumerate(qa_baselines):
        delta_report_path = _qa_delta_report_path(
            report_dir,
            candidate_name=qa_candidate_name,
            baseline_name=baseline_name,
            index=index,
            use_legacy_path=qa_baselines == ("majority",),
        )
        delta_report = qa_eval_delta_report(
            qa_candidate_report,
            qa_reports_by_agent[baseline_name],
            candidate_name=qa_candidate_name,
            baseline_name=baseline_name,
            candidate_report_path=qa_candidate_report_path,
            baseline_report_path=qa_report_paths[baseline_name],
        )
        save_qa_eval_delta_report(delta_report, delta_report_path)
        qa_delta_report_paths[baseline_name] = delta_report_path
    qa_delta_report_path = qa_delta_report_paths[primary_qa_baseline_name]
    qa_graph_construction_delta_report_path = (
        report_dir / "zz-qa-delta-graph-tool-vs-predicted-graph-tool.json"
    )
    qa_graph_construction_delta_report = qa_eval_delta_report(
        qa_candidate_report,
        predicted_graph_tool_report,
        candidate_name=qa_candidate_name,
        baseline_name=qa_graph_construction_baseline_name,
        candidate_report_path=qa_candidate_report_path,
        baseline_report_path=predicted_graph_tool_report_path,
    )
    save_qa_eval_delta_report(
        qa_graph_construction_delta_report,
        qa_graph_construction_delta_report_path,
    )
    qa_delta_report_paths[qa_graph_construction_baseline_name] = (
        qa_graph_construction_delta_report_path
    )

    active_task_path = active_dir / "active-tasks.jsonl"
    active_candidate_report_path = report_dir / "active-oracle-report.json"
    active_baseline_report_path = report_dir / "active-direct-report.json"
    active_predicted_graph_report_path = (
        report_dir / "active-predicted-graph-report.json"
    )
    active_delta_report_path = report_dir / "active-delta-report.json"
    active_graph_construction_delta_report_path = (
        report_dir / "zz-active-delta-oracle-vs-predicted-graph.json"
    )
    active_tasks = tuple(
        _active_task_for_case(group[0]) for group in qa_case_groups if group
    )
    active_graphs = tuple(
        graph for graph, group in zip(graphs, qa_case_groups, strict=True) if group
    )
    active_predicted_graphs = tuple(
        graph
        for graph, group in zip(predicted_graphs, qa_case_groups, strict=True)
        if group
    )
    save_active_eqa_tasks(active_tasks, active_task_path)
    active_candidate_agent = ActiveGraphAgent(policy="oracle_evidence")
    active_baseline_agent = ActiveGraphAgent(policy="direct_answer")
    active_candidate_results = tuple(
        active_candidate_agent.run(
            task,
            MockActiveEnvironment({task.initial_step: graph}),
        )
        for task, graph in zip(active_tasks, active_graphs, strict=True)
    )
    active_predicted_graph_results = tuple(
        active_candidate_agent.run(
            task,
            MockActiveEnvironment({task.initial_step: graph}),
        )
        for task, graph in zip(active_tasks, active_predicted_graphs, strict=True)
    )
    active_baseline_results = tuple(
        active_baseline_agent.run(
            task,
            MockActiveEnvironment(
                {
                    task.initial_step: _graph_without_required_object(
                        graph,
                        task,
                    )
                }
            ),
        )
        for task, graph in zip(active_tasks, active_graphs, strict=True)
    )
    active_report_graph_path = graph_paths[0] if len(graph_paths) == 1 else None
    active_predicted_report_graph_path = (
        predicted_graph_paths[0] if len(predicted_graph_paths) == 1 else None
    )
    active_candidate_report = active_task_report(
        active_tasks,
        active_candidate_results,
        task_path=active_task_path,
        graph_path=active_report_graph_path,
        policy="oracle_evidence",
    )
    active_baseline_report = active_task_report(
        active_tasks,
        active_baseline_results,
        task_path=active_task_path,
        graph_path=None,
        policy="direct_answer",
    )
    active_predicted_graph_report = active_task_report(
        active_tasks,
        active_predicted_graph_results,
        task_path=active_task_path,
        graph_path=active_predicted_report_graph_path,
        policy="oracle_evidence",
    )
    save_active_task_report(active_candidate_report, active_candidate_report_path)
    save_active_task_report(active_baseline_report, active_baseline_report_path)
    save_active_task_report(
        active_predicted_graph_report,
        active_predicted_graph_report_path,
    )
    active_delta = active_task_delta_report(
        active_candidate_report,
        active_baseline_report,
        candidate_name="oracle_evidence",
        baseline_name="direct_answer",
        candidate_report_path=active_candidate_report_path,
        baseline_report_path=active_baseline_report_path,
    )
    save_active_task_delta_report(active_delta, active_delta_report_path)
    active_graph_construction_delta = active_task_delta_report(
        active_candidate_report,
        active_predicted_graph_report,
        candidate_name="oracle_evidence",
        baseline_name="predicted_graph_evidence",
        candidate_report_path=active_candidate_report_path,
        baseline_report_path=active_predicted_graph_report_path,
    )
    save_active_task_delta_report(
        active_graph_construction_delta,
        active_graph_construction_delta_report_path,
    )

    manifest = build_benchmark_artifacts(
        dataset_name=dataset_name,
        episode_paths=tuple(episode_paths),
        output_dir=benchmark_dir,
        max_qa_per_episode=max_qa_per_episode,
        qa_eval_delta_report_paths=tuple(qa_delta_report_paths.values()),
        active_task_delta_report_paths=(
            active_delta_report_path,
            active_graph_construction_delta_report_path,
        ),
        error_attribution_report_paths=tuple(error_attribution_report_paths),
        graph_eval_report_paths=tuple(graph_eval_report_paths),
        predicted_graph_report_paths=tuple(predicted_graph_report_paths),
    )
    save_benchmark_manifest(manifest, manifest_path)
    summary_report = experiment_summary_report(manifest, manifest_path=manifest_path)
    save_experiment_summary_report(summary_report, summary_report_path)

    dashboard = dashboard_bundle(
        qa_cases,
        predictions=graph_tool_predictions,
        qa_eval_report=qa_candidate_report,
        graph=graphs[0],
        error_attribution_report=(
            error_attribution_reports[0] if error_attribution_reports else None
        ),
        active_task_report=active_candidate_report,
        active_task_delta_report=active_delta,
        experiment_summary_report=summary_report,
    )
    dashboard_result = export_dashboard(dashboard, dashboard_dir)
    record = experiment_record(
        summary_report,
        summary_report_path=summary_report_path,
        dashboard_bundle=dashboard,
        dashboard_bundle_path=dashboard_result["bundle_path"],
    )
    save_experiment_record(record, record_path)

    return {
        "schema_version": MOCK_EXPERIMENT_RESULT_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "output_dir": str(root),
        "episode_path": str(episode_paths[0]),
        "episode_paths": [str(path) for path in episode_paths],
        "graph_path": str(graph_paths[0]),
        "graph_paths": [str(path) for path in graph_paths],
        "predicted_graph_path": str(predicted_graph_paths[0]),
        "predicted_graph_paths": [str(path) for path in predicted_graph_paths],
        "predicted_graph_report_path": str(predicted_graph_report_paths[0]),
        "predicted_graph_report_paths": [
            str(path) for path in predicted_graph_report_paths
        ],
        "graph_eval_report_path": str(graph_eval_report_paths[0]),
        "graph_eval_report_paths": [str(path) for path in graph_eval_report_paths],
        "error_attribution_report_path": str(error_attribution_report_paths[0]),
        "error_attribution_report_paths": [
            str(path) for path in error_attribution_report_paths
        ],
        "qa_path": str(qa_paths[0]),
        "qa_paths": [str(path) for path in qa_paths],
        "combined_qa_path": str(combined_qa_path),
        "qa_candidate_name": qa_candidate_name,
        "qa_baseline_names": list(qa_baselines),
        "qa_graph_construction_baseline_name": qa_graph_construction_baseline_name,
        "qa_prediction_paths": {
            name: str(path) for name, path in qa_prediction_paths.items()
        },
        "qa_report_paths": {
            name: str(path) for name, path in qa_report_paths.items()
        },
        "qa_delta_report_paths": {
            name: str(path) for name, path in qa_delta_report_paths.items()
        },
        "graph_tool_prediction_path": str(graph_tool_prediction_path),
        "predicted_graph_tool_prediction_path": str(
            predicted_graph_tool_prediction_path
        ),
        "majority_prediction_path": str(majority_prediction_path),
        "qa_candidate_report_path": str(qa_candidate_report_path),
        "qa_baseline_report_path": str(qa_baseline_report_path),
        "predicted_graph_tool_report_path": str(predicted_graph_tool_report_path),
        "qa_delta_report_path": str(qa_delta_report_path),
        "qa_graph_construction_delta_report_path": str(
            qa_graph_construction_delta_report_path
        ),
        "active_task_path": str(active_task_path),
        "active_candidate_report_path": str(active_candidate_report_path),
        "active_baseline_report_path": str(active_baseline_report_path),
        "active_predicted_graph_report_path": str(active_predicted_graph_report_path),
        "active_delta_report_path": str(active_delta_report_path),
        "active_graph_construction_delta_report_path": str(
            active_graph_construction_delta_report_path
        ),
        "manifest_path": str(manifest_path),
        "summary_report_path": str(summary_report_path),
        "dashboard_bundle_path": str(dashboard_result["bundle_path"]),
        "dashboard_index_path": str(dashboard_result["index_path"]),
        "record_path": str(record_path),
        "readiness_status": record["readiness_status"],
        "verdict_counts": record["verdict_counts"],
        "record_digest": record["record_digest"],
    }


def _qa_baseline_names(qa_baseline_names: Sequence[str] | None) -> tuple[str, ...]:
    if qa_baseline_names is None:
        return ("majority",)
    names = tuple(str(name) for name in qa_baseline_names)
    if not names:
        raise ValueError("Mock experiment requires at least one QA baseline")
    for name in names:
        if name == "":
            raise ValueError("Mock experiment QA baseline names must be non-empty")
        if name == "graph_tool":
            raise ValueError("Mock experiment QA baseline cannot be graph_tool")
    if len(set(names)) != len(names):
        raise ValueError("Mock experiment QA baseline names must be unique")
    return names


def _qa_delta_report_path(
    report_dir: Path,
    *,
    candidate_name: str,
    baseline_name: str,
    index: int,
    use_legacy_path: bool,
) -> Path:
    if use_legacy_path:
        return report_dir / "qa-delta-report.json"
    return (
        report_dir
        / (
            f"qa-delta-{index:02d}-{_agent_slug(candidate_name)}"
            f"-vs-{_agent_slug(baseline_name)}.json"
        )
    )


def _agent_slug(name: str) -> str:
    return _path_stem(name).replace("_", "-")


def _episode_configs(
    config_type: type["AI2ThorAdapterConfig"],
    episode_configs: Sequence["AI2ThorAdapterConfig"] | None,
) -> tuple["AI2ThorAdapterConfig", ...]:
    if episode_configs is None:
        return (
            config_type(
                scene_id="FloorPlan1",
                episode_id="ai2thor_mock_001",
                steps=(1, 2),
                actions=("Initialize", "MoveAhead"),
            ),
        )
    configs = tuple(episode_configs)
    if not configs:
        raise ValueError("Mock experiment requires at least one episode config")
    episode_ids = [config.episode_id for config in configs]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Mock experiment episode configs must have unique episode_id values")
    return configs


def _episode_path(
    root: Path,
    episode_id: str,
    *,
    use_legacy_single_episode_path: bool,
) -> Path:
    if use_legacy_single_episode_path:
        return root / "episodes" / "mock-episode.jsonl"
    return root / "episodes" / f"{_path_stem(episode_id)}.jsonl"


def _artifact_items(
    manifest: Mapping[str, Any],
    *,
    expected_count: int,
) -> tuple[Mapping[str, Any], ...]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, str):
        raise ValueError("Mock experiment benchmark artifacts must be a sequence")
    if len(artifacts) != expected_count:
        raise ValueError("Mock experiment benchmark artifact count must match episode count")
    items: list[Mapping[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("Mock experiment artifact must be an object")
        items.append(item)
    return tuple(items)


def _required_artifact_path(artifact: Mapping[str, Any], key: str) -> Path:
    value = artifact.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"Mock experiment artifact {key} must be a non-empty path")
    return Path(value)


def _episode_id_from_frames(frames: Sequence[Any]) -> str:
    if not frames:
        raise ValueError("Mock experiment predicted graph requires episode frames")
    episode_id = getattr(frames[0], "episode_id", "")
    if not isinstance(episode_id, str) or episode_id == "":
        raise ValueError("Mock experiment episode frame requires a non-empty episode_id")
    return episode_id


def _path_stem(value: str) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return stem or "mock_experiment"


def _active_task_for_case(case: Any) -> ActiveEQATask:
    return ActiveEQATask(
        id=f"active:{case.id}",
        scene_id=case.scene_id,
        episode_id=case.episode_id,
        initial_step=case.step,
        question=case.question,
        gold_answer=case.answer,
        success_conditions={"answer_exact": True},
        max_actions=1,
        required_evidence={
            "nodes": case.required_nodes,
            "edges": case.required_edges,
        },
    )


def _graph_without_required_object(
    graph: DynamicSceneGraph,
    task: ActiveEQATask,
) -> DynamicSceneGraph:
    required_nodes = tuple(task.required_evidence.get("nodes", ()))
    object_id = required_nodes[0] if required_nodes else ""
    cloned = graph_from_json(graph_to_json(graph))
    if object_id == "":
        return cloned
    cloned.object_states.pop(object_id, None)
    cloned.object_state_history.pop(object_id, None)
    for node_id in list(cloned.nodes):
        if node_id == object_id or node_id.startswith(f"state:{object_id}:"):
            cloned.nodes.pop(node_id)
    cloned.edges = [
        edge
        for edge in cloned.edges
        if edge.src != object_id
        and edge.dst != object_id
        and not edge.dst.startswith(f"state:{object_id}:")
    ]
    return cloned
