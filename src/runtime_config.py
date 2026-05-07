import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_CONFIG = {
    "brief_source": "data/raw/assessment_brief.txt",
    "supporting_sources": ["data/raw/project_design.txt"],
}


def _config_path(config_path=None):
    """Return the runtime configuration file path."""
    if config_path is not None:
        return Path(config_path)
    return PROJECT_ROOT / "data" / "runtime_sources.json"


def load_runtime_config(config_path=None):
    """Load the runtime source configuration with safe defaults."""
    runtime_config = dict(DEFAULT_RUNTIME_CONFIG)
    config_file = _config_path(config_path)

    if not config_file.exists():
        return runtime_config

    with open(config_file, "r", encoding="utf-8") as input_file:
        loaded_config = json.load(input_file)

    if isinstance(loaded_config.get("brief_source"), str) and loaded_config["brief_source"].strip():
        runtime_config["brief_source"] = loaded_config["brief_source"].strip()

    supporting_sources = loaded_config.get("supporting_sources", [])
    if isinstance(supporting_sources, list):
        runtime_config["supporting_sources"] = [
            str(source).strip()
            for source in supporting_sources
            if str(source).strip()
        ]

    return runtime_config


def resolve_runtime_path(path_value):
    """Resolve a configured relative or absolute path against the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_active_brief_path(config_path=None):
    """Return the currently configured authoritative assignment brief path."""
    runtime_config = load_runtime_config(config_path=config_path)
    return resolve_runtime_path(runtime_config["brief_source"])


def get_supporting_source_paths(config_path=None):
    """Return any configured project-side supporting sources."""
    runtime_config = load_runtime_config(config_path=config_path)
    return [resolve_runtime_path(path_value) for path_value in runtime_config.get("supporting_sources", [])]


def iter_runtime_sources(config_path=None):
    """Yield the active brief first, then any supporting project sources."""
    yield {
        "path": get_active_brief_path(config_path=config_path),
        "authority": "official",
        "role": "brief",
    }

    for source_path in get_supporting_source_paths(config_path=config_path):
        yield {
            "path": source_path,
            "authority": "project",
            "role": "supporting",
        }
