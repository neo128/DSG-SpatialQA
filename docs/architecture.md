# DSG-SpatialQA Lab Architecture

This lab is a deterministic validation platform for testing whether Dynamic
Scene Graphs improve spatial QA, dynamic memory, `GraphTool` queries, and
interactive task ability.

The default runtime is local and in memory. It does not call live AI services,
robot stacks, simulators, clocks, or unstable sources. Callers supply explicit
episode steps, graph files, task files, and report paths.

## Validation Loop

The implemented MVP is organized as a repeatable artifact chain:

```text
explicit episode JSONL
-> real collection evidence report
-> oracle Dynamic Scene Graph
-> predicted/mock Dynamic Scene Graph
-> predicted DSG evidence report
-> QA JSONL and active task JSONL
-> local baselines and graph-tool answers
-> QA, graph, and active-task metrics
-> error attribution
-> dashboard bundle and benchmark manifest
-> real experiment package assembly
-> real experiment readiness report
```

Each stage has explicit local inputs and explicit local outputs. Most stages
also provide validation or comparison helpers so saved artifacts can be checked
against current code without calling external services.

## Research Questions

### RQ1: Spatial QA

Spatial QA is evaluated by comparing oracle DSG answers, predicted DSG answers,
and local baselines over the same QA cases.

Current evidence:

- exact-match and multiple-choice QA accuracy,
- numeric MAE for metric answers,
- evidence node and edge recall,
- question type, tag, and reference-frame breakdowns,
- benchmark manifests that aggregate coverage across episodes.

Primary modules and scripts:

- `dsg_spatialqa_lab.benchmark.qa_generator`
- `dsg_spatialqa_lab.eval.qa_metrics`
- `dsg_spatialqa_lab.agents.graph_tool_agent`
- `scripts/generate_qa.py`
- `scripts/run_baselines.py`
- `scripts/run_qa_eval.py`

### RQ2: Dynamic Memory

Dynamic memory is evaluated through explicit-step histories, object timelines,
relation timelines, scene deltas, recent events, last-seen state, and
re-observation targets.

Current evidence:

- built-in evaluation cases for object and agent timelines,
- episode-derived object state history,
- moved-object and low-confidence hidden-object diagnostics,
- QA cases whose answers depend on temporal state.

Primary modules and scripts:

- `dsg_spatialqa_lab.memory`
- `dsg_spatialqa_lab.graph_tool`
- `dsg_spatialqa_lab.oracle`
- `scripts/evaluate.py`

### RQ3: GraphTool Query Utility

`GraphTool` utility is evaluated by comparing direct graph-tool answers with
graph-text, majority, disabled caption-memory placeholder, and future offline
external-model predictions.

Current evidence:

- `graph_query` QA intent,
- `retrieve_subgraph` QA intent,
- local `graph_tool` baseline,
- error attribution that separates graph construction failures from
  reasoning/tool-use failures.

Primary modules and scripts:

- `dsg_spatialqa_lab.graph_tool`
- `dsg_spatialqa_lab.qa`
- `dsg_spatialqa_lab.eval.error_attribution`
- `scripts/attribute_errors.py`

### RQ4: Interactive Task Ability

Interactive ability is represented by deterministic mock active EQA tasks. The
mock environment advances only through caller-supplied graph steps.

Current evidence:

- task success,
- answer accuracy,
- action count,
- evidence coverage,
- answer-graph consistency.

Primary modules and scripts:

- `dsg_spatialqa_lab.tasks.active_eqa`
- `dsg_spatialqa_lab.agents.active_graph_agent`
- `dsg_spatialqa_lab.eval.task_metrics`
- `scripts/run_active_tasks.py`

## Component Map

| Layer | Current implementation | Boundary |
| --- | --- | --- |
| Scene graph core | `DynamicSceneGraph`, `GraphTool`, relation engine, QA engine | In memory only |
| Episode input | Episode JSONL with explicit steps | No simulator needed |
| Real collection evidence | AI2-THOR/Habitat episode collection gate | Explicit local episode files only |
| Oracle graph | Episode metadata to DSG | Uses explicit metadata |
| Predicted graph | Mock detections or explicit observation sequences to DSG with source metadata | No real perception model |
| Baselines | `graph_tool`, `majority`, `graph_text`, disabled `caption_memory` | No model calls |
| Offline prediction imports | Local external prediction JSONL to `QAPrediction` JSONL | Explicit files only |
| Offline control matrix | VLM-only, multi-frame VLM, caption-memory, graph-text import gate | Explicit import reports only |
| Predicted DSG evidence | Observation-sequence RGB-D/detector evidence gate | Explicit predicted graph reports and observation files only |
| QA metrics | QA prediction JSONL reports | Explicit gold/pred paths |
| Graph metrics | Oracle-vs-predicted graph reports | Exact by default; label+center optional |
| Error attribution | Oracle, predicted, and prediction comparison | Explicit local files |
| Dashboard | Static JSON/HTML bundle | QA/error review plus optional active panels |
| Active tasks | Mock graph-step EQA loop | No navigation or robot control |
| Benchmark | Multi-episode manifest and coverage | Explicit episode paths |
| Real experiment package assembly | Manifest plus readiness report writer | Explicit local artifacts only |
| Real experiment readiness | Manifest-linked evidence gate | Explicit local artifacts only |
| Adapters | AI2-THOR and Habitat mock collectors | Real collection fails closed |

## Deterministic Runtime Rules

- No runtime network calls.
- No wall-clock reads.
- No unstable output sources.
- No default simulator, VLM, LLM, robot, or perception dependencies.
- Every artifact path is supplied by the caller.
- JSON and JSONL outputs use stable ordering and digest helpers.
- Optional integrations must live in adapter layers and fail closed when the
  dependency is absent.

## What Is Implemented

The deterministic validation-loop MVP is implemented through benchmark manifest
tooling:

```text
mock AI2-THOR or Habitat episode
-> oracle graph
-> predicted/mock graph
-> generated QA
-> local baseline predictions
-> QA and graph metrics
-> error attribution
-> dashboard bundle
-> active task report
-> benchmark manifest
```

This is sufficient to run local, reproducible experiments over small mock
episodes and scene fixtures.

## What Remains Outside The MVP

The following are intentionally not part of the default deterministic runtime:

- real VLM or LLM predictions,
- real simulator collection,
- real robot interaction,
- real navigation or next-best-view planning,
- real perception model execution,
- persistence layers or databases,
- advanced graph matching beyond exact and label+center modes, such as
  room-aware and confidence-weighted matching.

Future work should add these only behind optional, explicit, mock-first
boundaries. Detector or RGB-D pipelines should first write local
`SceneObservation` sequence artifacts, then feed those artifacts through the
predicted graph builder for deterministic graph, QA, attribution, manifest, and
dashboard evaluation.
