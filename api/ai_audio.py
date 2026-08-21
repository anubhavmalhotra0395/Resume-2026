"""
AI Audio module — translates natural language prompts into DSP configuration
using GPT-4 (OpenAI).  Keeps state between iterative calls so users can
refine their sound conversationally.

DSP config schema (strict):
{
  "eq": {
    "low_shelf_db": float,       // sub + low body  (<250 Hz)
    "low_mid_db": float,         // mud / warmth    (250-500 Hz)
    "mid_db": float,             // presence        (500-2000 Hz)
    "high_mid_db": float,        // clarity / edge  (2000-6000 Hz)
    "high_shelf_db": float       // air             (>6 kHz)
  },
  "compression": {
    "threshold_db": float,       // e.g. -20
    "ratio": float,              // 1 - 20
    "attack_ms": float,          // 0.1 - 100
    "release_ms": float,         // 10 - 500
    "makeup_db": float           // 0 - 12
  },
  "reverb": {
    "decay_s": float,            // 0.2 - 3.0
    "mix": float,                // 0.0 - 0.3
    "pre_delay_ms": float        // 0 - 50
  },
  "saturation": {
    "drive": float               // 1.0 - 3.0 (1.0 = off)
  },
  "width": {
    "mix": float,                // 0.0 - 0.6
    "delay_ms": float,           // 8 - 20
    "detune_cents": float        // 2 - 10
  }
}
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded sane defaults — used as fallback if GPT call fails
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "eq": {
        "low_shelf_db": 0.0,
        "low_mid_db": 0.0,
        "mid_db": 0.0,
        "high_mid_db": 0.0,
        "high_shelf_db": 0.0,
    },
    "compression": {
        "threshold_db": -20.0,
        "ratio": 3.0,
        "attack_ms": 10.0,
        "release_ms": 80.0,
        "makeup_db": 4.0,
    },
    "reverb": {
        "decay_s": 0.8,
        "mix": 0.15,
        "pre_delay_ms": 15.0,
    },
    "saturation": {
        "drive": 1.0,
    },
    "width": {
        "mix": 0.35,
        "delay_ms": 12.0,
        "detune_cents": 4.0,
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """
You are an expert audio mixing engineer and DSP specialist.
Your job is to convert user descriptions about how they want their audio to sound
into a precise JSON DSP configuration. You ONLY output valid JSON. No explanations.
No markdown. No extra text. Just the raw JSON object.

The JSON must strictly follow this schema:
{
  "eq": {
    "low_shelf_db": <float -12 to 12>,
    "low_mid_db": <float -12 to 12>,
    "mid_db": <float -12 to 12>,
    "high_mid_db": <float -12 to 12>,
    "high_shelf_db": <float -12 to 12>
  },
  "compression": {
    "threshold_db": <float -40 to -6>,
    "ratio": <float 1 to 20>,
    "attack_ms": <float 0.1 to 100>,
    "release_ms": <float 10 to 500>,
    "makeup_db": <float 0 to 12>
  },
  "reverb": {
    "decay_s": <float 0.2 to 3.0>,
    "mix": <float 0.0 to 0.30>,
    "pre_delay_ms": <float 0 to 50>
  },
  "saturation": {
    "drive": <float 1.0 to 3.0>
  },
  "width": {
    "mix": <float 0.0 to 0.6>,
    "delay_ms": <float 8 to 20>,
    "detune_cents": <float 2 to 10>
  }
}

Translate human descriptive terms to DSP values using this guide:
- warm / warmer           → boost low_shelf (+2..+4), cut high_mid (-1..-2)
- harsh / bright          → cut high_mid (-2..-4), cut high_shelf (-1..-2)
- muddy                   → cut low_mid (-3..-5)
- thin / weak             → boost low_shelf (+2..+3), boost mid (+1..+2)
- punchy / tight          → lower threshold (-24..-28), ratio 4-6, fast attack (5-15ms)
- airy / open             → boost high_shelf (+2..+3)
- presence / forward      → boost mid (+1..+3), boost high_mid (+1..+2)
- nasal                   → cut mid (-2..-4) around 1-2kHz
- boxy                    → cut low_mid (-3) around 300-500Hz
- echo / reverb           → increase reverb mix (0.2-0.3), increase decay
- dry / less reverb       → decrease reverb mix (0.0-0.08)
- wide / spacious         → increase width mix (0.4-0.6)
- mono / centered         → decrease width mix (0.0-0.1)
- grit / bite / edgy      → increase saturation drive (1.5-2.5)
- clean / smooth          → saturation drive 1.0
- more punch / transients → lower threshold, fast attack (1-5ms), ratio 4-8

Always keep reverb mix ≤ 0.30. Always keep saturation drive ≤ 3.0.
When updating a previous config, only change the fields that the new prompt affects.
Return ONLY the complete updated JSON object.
""".strip()


# ---------------------------------------------------------------------------
# JSON validation / coercion
# ---------------------------------------------------------------------------
def _clamp(val: Any, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(val))))
    except (TypeError, ValueError):
        return (lo + hi) / 2.0


def validate_dsp_config(raw: dict) -> dict:
    """Validate and clamp all DSP config values to safe ranges."""
    eq = raw.get("eq", {})
    comp = raw.get("compression", {})
    rev = raw.get("reverb", {})
    sat = raw.get("saturation", {})
    wid = raw.get("width", {})

    return {
        "eq": {
            "low_shelf_db": _clamp(eq.get("low_shelf_db", 0), -12, 12),
            "low_mid_db": _clamp(eq.get("low_mid_db", 0), -12, 12),
            "mid_db": _clamp(eq.get("mid_db", 0), -12, 12),
            "high_mid_db": _clamp(eq.get("high_mid_db", 0), -12, 12),
            "high_shelf_db": _clamp(eq.get("high_shelf_db", 0), -12, 12),
        },
        "compression": {
            "threshold_db": _clamp(comp.get("threshold_db", -20), -40, -6),
            "ratio": _clamp(comp.get("ratio", 3), 1, 20),
            "attack_ms": _clamp(comp.get("attack_ms", 10), 0.1, 100),
            "release_ms": _clamp(comp.get("release_ms", 80), 10, 500),
            "makeup_db": _clamp(comp.get("makeup_db", 4), 0, 12),
        },
        "reverb": {
            "decay_s": _clamp(rev.get("decay_s", 0.8), 0.2, 3.0),
            "mix": _clamp(rev.get("mix", 0.15), 0.0, 0.30),
            "pre_delay_ms": _clamp(rev.get("pre_delay_ms", 15), 0, 50),
        },
        "saturation": {
            "drive": _clamp(sat.get("drive", 1.0), 1.0, 3.0),
        },
        "width": {
            "mix": _clamp(wid.get("mix", 0.35), 0.0, 0.6),
            "delay_ms": _clamp(wid.get("delay_ms", 12), 8, 20),
            "detune_cents": _clamp(wid.get("detune_cents", 4), 2, 10),
        },
    }


# ---------------------------------------------------------------------------
# Conversation session state (in-memory; keyed by session_id)
# ---------------------------------------------------------------------------
_sessions: dict[str, list[dict]] = {}


def _get_history(session_id: Optional[str]) -> list[dict]:
    if not session_id:
        return []
    return _sessions.get(session_id, [])


def _save_history(session_id: Optional[str], history: list[dict]) -> None:
    if session_id:
        _sessions[session_id] = history[-20:]  # keep last 20 turns


# ---------------------------------------------------------------------------
# Core function: prompt → DSP config
# ---------------------------------------------------------------------------
def prompt_to_dsp_config(
    prompt: str,
    audio_features: Optional[dict] = None,
    previous_config: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Convert a natural language prompt into a validated DSP config dict.

    Args:
        prompt: User's description, e.g. "Make vocals warmer, less harsh"
        audio_features: Optional extracted audio features (loudness, eq, dynamics)
        previous_config: Previous DSP config for iterative updates
        session_id: Optional session key for conversational continuity

    Returns:
        Validated DSP config dict (always safe to use even if GPT fails)
    """
    # Provider priority: GROQ_API_KEY → OPENAI_API_KEY → fallback defaults
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    api_key = groq_key or openai_key
    use_groq = bool(groq_key)

    if not api_key:
        logger.warning("No API key set (GROQ_API_KEY or OPENAI_API_KEY) — returning defaults")
        base = previous_config or _DEFAULTS.copy()
        return validate_dsp_config(base)

    try:
        from openai import OpenAI  # lazy import; graceful if not installed
        if use_groq:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            model = "llama-3.3-70b-versatile"
        else:
            client = OpenAI(api_key=api_key)
            model = "gpt-4o"
    except ImportError:
        logger.error("openai package not installed — returning defaults")
        base = previous_config or _DEFAULTS.copy()
        return validate_dsp_config(base)

    # Build message history
    history = _get_history(session_id)

    # Compose the user message
    user_parts = [f'User request: "{prompt}"']

    if audio_features:
        user_parts.append(
            f"Current audio features:\n{json.dumps(audio_features, indent=2)}"
        )

    if previous_config:
        user_parts.append(
            f"Previous DSP config (update only what's needed):\n"
            f"{json.dumps(previous_config, indent=2)}"
        )
    else:
        user_parts.append("No previous config — generate a fresh config.")

    user_message = "\n\n".join(user_parts)

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=512,
        )

        raw_text = response.choices[0].message.content or "{}"
        logger.debug("GPT raw response: %s", raw_text)

        raw_json = json.loads(raw_text)
        config = validate_dsp_config(raw_json)

        # Save turn to history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": raw_text})
        _save_history(session_id, history)

        return config

    except json.JSONDecodeError as exc:
        logger.error("GPT returned invalid JSON: %s", exc)
        base = previous_config or _DEFAULTS.copy()
        return validate_dsp_config(base)
    except Exception as exc:
        logger.error("OpenAI call failed: %s", exc)
        base = previous_config or _DEFAULTS.copy()
        return validate_dsp_config(base)


# ---------------------------------------------------------------------------
# Reference track matching helper
# ---------------------------------------------------------------------------
def reference_match_config(
    reference_features: dict,
    current_features: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Generate DSP settings to match a reference track's characteristics.

    Args:
        reference_features: Extracted features from the reference track
        current_features: Optional features from the current (dry) track
        session_id: Optional session ID

    Returns:
        Validated DSP config dict
    """
    parts = [
        "Generate DSP settings to match the reference vocal characteristics.",
        f"Reference audio features:\n{json.dumps(reference_features, indent=2)}",
    ]
    if current_features:
        parts.append(
            f"Current (dry) audio features:\n{json.dumps(current_features, indent=2)}"
        )
    parts.append(
        "Create a config that transforms the dry audio to match the reference style."
    )

    prompt = "\n\n".join(parts)
    return prompt_to_dsp_config(prompt, session_id=session_id)


# ---------------------------------------------------------------------------
# Smart preset from audio features
# ---------------------------------------------------------------------------
def generate_preset_from_features(
    features: dict,
    style_hint: str = "optimal vocal processing",
    session_id: Optional[str] = None,
) -> dict:
    """
    Generate an optimal preset based on extracted audio features.

    Args:
        features: Extracted audio features dict (loudness, spectral info, dynamics)
        style_hint: Human style description, e.g. "radio-ready pop vocal"
        session_id: Optional session ID

    Returns:
        Validated DSP config dict
    """
    prompt = (
        f'Based on these audio features, suggest a "{style_hint}" processing chain.\n\n'
        f"Audio features:\n{json.dumps(features, indent=2)}"
    )
    return prompt_to_dsp_config(prompt, audio_features=features, session_id=session_id)


# ---------------------------------------------------------------------------
# Clear session
# ---------------------------------------------------------------------------
def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
