import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL_HOST = "generativelanguage.googleapis.com"
GEMINI_CONSERVATIVE_FREE_MODE_RPM = {
    "gemini-2.5-flash-lite": 8,
    "gemini-2.5-flash": 4,
    "gemini-3.1-flash-lite": 12,
    "gemini-3-flash": 4,
}

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _mask_secret(secret):
    """Return a masked representation of an API key for safe logging."""
    if not secret:
        return "missing"
    if len(secret) <= 8:
        return "***present***"
    return f"{secret[:4]}...{secret[-4:]}"


def _guess_key_provider(secret):
    """Infer the likely provider family from the key prefix."""
    if not secret:
        return "missing"
    if secret.startswith("sk-") or secret.startswith("sess-"):
        return "openai-like"
    if secret.startswith("AIza"):
        return "google-like"
    return "unknown"


def _normalize_base_url(base_url):
    """Return the configured base URL or the default OpenAI endpoint."""
    if not base_url:
        return DEFAULT_BASE_URL
    return base_url.rstrip("/")


def _free_mode_is_enabled():
    """Return True when conservative free-tier throttling is enabled."""
    return os.getenv("LLM_FREE_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _free_mode_rpm_hint(model, base_url):
    """Return the conservative RPM hint applied to Gemini models in free mode."""
    normalized_model = model.strip().lower()
    normalized_base_url = base_url.strip().lower()
    if not _free_mode_is_enabled():
        return None
    if not (normalized_model.startswith("gemini") or GEMINI_BASE_URL_HOST in normalized_base_url):
        return None

    if normalized_model in GEMINI_CONSERVATIVE_FREE_MODE_RPM:
        return GEMINI_CONSERVATIVE_FREE_MODE_RPM[normalized_model]

    for known_model, rpm_cap in GEMINI_CONSERVATIVE_FREE_MODE_RPM.items():
        if normalized_model.startswith(known_model):
            return rpm_cap

    return 4


def _auth_result_label(exc):
    """Convert an API exception into a short human-readable status label."""
    status_code = getattr(exc, "status_code", None)
    error = getattr(exc, "body", None)
    if status_code == 401:
        return "failed (401 invalid or unauthorized key)"
    if status_code == 403:
        return "failed (403 forbidden or project restriction)"
    if status_code == 404:
        return "failed (404 model or endpoint not found)"
    if error:
        return f"failed ({error})"
    return f"failed ({exc.__class__.__name__})"


def build_openai_self_check(run_remote_check=False):
    """Build a masked self-check report for the configured LLM provider."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = _normalize_base_url(os.getenv("OPENAI_BASE_URL", "").strip())

    report = {
        "api_key_present": bool(api_key),
        "api_key_masked": _mask_secret(api_key),
        "api_key_provider_guess": _guess_key_provider(api_key),
        "model": model,
        "base_url": base_url,
        "free_mode_enabled": _free_mode_is_enabled(),
        "free_mode_rpm_hint": _free_mode_rpm_hint(model, base_url),
        "remote_check": "skipped",
        "hint": None,
    }

    if not api_key:
        report["hint"] = "OPENAI_API_KEY is missing from the runtime environment."
        return report

    if report["api_key_provider_guess"] == "google-like" and base_url == DEFAULT_BASE_URL:
        report["hint"] = (
            "The key looks Google/Gemini-style, but the app is using the default OpenAI API base URL."
        )

    if not run_remote_check:
        return report

    if OpenAI is None:
        report["remote_check"] = "failed (openai package missing)"
        return report

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        client.models.retrieve(model)
        report["remote_check"] = "passed"
    except Exception as exc:
        report["remote_check"] = _auth_result_label(exc)

    return report


def format_openai_self_check(report):
    """Format the self-check report as compact startup text."""
    parts = [
        f"key={report['api_key_masked']}",
        f"provider_guess={report['api_key_provider_guess']}",
        f"model={report['model']}",
        f"base_url={report['base_url']}",
        f"free_mode={report['free_mode_enabled']}",
        f"remote_check={report['remote_check']}",
    ]
    if report.get("free_mode_rpm_hint") is not None:
        parts.insert(4, f"free_mode_rpm={report['free_mode_rpm_hint']}")
    line = "LLM Check: " + ", ".join(parts)
    if report.get("hint"):
        line += f"\nHint: {report['hint']}"
    return line
