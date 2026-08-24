from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", case_sensitive=False)
    storage_root: Path = Path("storage")
    inputs_dir: Path = Path("storage/inputs")
    outputs_dir: Path = Path("storage/outputs")
    redis_url: str = "redis://localhost:6379/0"
    loudness_target_lufs: float = -16.0
    sample_rate: int = 44100
    delete_after_hours: int = 24
    # Hard cap on both uploads (reference and dry vocal). Separation cost and
    # peak RAM both scale with duration, so this is a real resource guard.
    max_duration_seconds: int = 240
    max_file_mb: int = 50
    # Reference audio beyond this is not analysed (the reference only feeds
    # analysis — none of it ends up in the output, so 2 minutes is plenty).
    analysis_max_seconds: int = 120
    # Window used by /analyze-layers. Layering is a local property, so a
    # single well-chosen window (the chorus) is as informative as the whole
    # song and far cheaper to separate + analyse. The UI's waveform selector
    # mirrors this cap — keep them in sync (MAX_REF_SELECT_S in index.html).
    layer_analysis_window_s: int = 30
    # Plain str, not List: pydantic-settings JSON-decodes env vars for complex
    # fields BEFORE validators run, so APP_CORS_ORIGINS=* crashed at startup
    # on the first real deploy. Split via the property below.
    cors_origins: str = "*"
    frontend_dir: Path = Path("frontend")
    # Optional: https://www.football-data.org/ — Chelsea snapshot at GET /api/chelsea/football
    football_data_api_token: str | None = None

    @field_validator("inputs_dir", "outputs_dir", mode="before")
    @classmethod
    def _make_path(cls, v, info):  # noqa: N805
        path = Path(v)
        if path.is_absolute():
            return path
        root = Path((info.data or {}).get("storage_root", "storage"))
        return root / path

    @property
    def cors_origin_list(self) -> List[str]:
        return [s.strip() for s in self.cors_origins.split(",") if s.strip()] or ["*"]


settings = Settings()
settings.storage_root.mkdir(parents=True, exist_ok=True)
settings.inputs_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)

