import json
from pathlib import Path
import sys


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import audit_trajectory_coverage as audit_script  # noqa: E402


def test_load_audit_cases_accepts_active_qa_v2_records(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa-active.jsonl"
    qa_path.write_text(
        json.dumps(
            {
                "schema_version": "dsg-spatialqa-lab.active-qa-case.v2",
                "id": "case-1",
                "episode_id": "episode-006",
                "scene_id": "FloorPlan202",
                "question_type": "support_relation",
                "target": {"object_id": "mug_1"},
                "answer": {"relation": "ON", "dst": "table_1"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = audit_script._load_audit_cases(qa_path)  # noqa: SLF001

    assert len(cases) == 1
    assert cases[0].episode_id == "episode-006"
    assert cases[0].question == {"object_id": "mug_1"}
    assert cases[0].answer == {"current_location": {"relation": "ON", "dst": "table_1"}}
