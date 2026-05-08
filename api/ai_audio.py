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


# ---------------------------------------------------------------------------
# Iterative Refinement — second-pass correction EQ
# ---------------------------------------------------------------------------

_REFINE_SYSTEM_PROMPT = """
You are a mastering engineer doing a second-pass correction after the first
vocal processing attempt. You will receive:
  1. A per-band spectral distance between the processed output and the reference
  2. The output's spectral profile vs the reference's profile

Your job is to output ONLY a correction EQ as strict JSON — a list of parametric
bands that would bring the output closer to the reference.

Output format (ONLY valid JSON, no markdown):
{
  "correction_eq": [
    {"f": <Hz float>, "gain_db": <float -6 to +6>, "q": <float 0.5 to 2.0>},
    ...
  ],
  "correction_summary": "<one sentence describing what the output needs>"
}

Rules:
- Output 4–8 bands maximum
- Only correct bands where there is a meaningful gap (>1 dB)
- Keep all gain_db values within ±5 dB
- If the output already sounds very close to reference (all gaps < 0.5 dB), return an empty correction_eq list
- Focus on the most audible frequency ranges: 100–8000 Hz
- Boost where the reference has more energy; cut where the output has too much
""".strip()


def ai_refinement_eq(
    output_profile: dict,
    reference_profile: dict,
    spectral_distance: float,
) -> dict:
    """
    Generate a correction EQ to reduce the spectral distance between
    the processed output and the reference vocal on a second pass.

    Args:
        output_profile:    Band energy profile of the processed output
        reference_profile: Band energy profile of the reference vocal
        spectral_distance: Overall spectral distance (0–1, lower is better)

    Returns:
        Dict with keys: correction_eq (List of band dicts), correction_summary
    """
    _safe = {"correction_eq": [], "correction_summary": "No correction needed."}

    # If already very close, skip the LLM call entirely
    if spectral_distance < 0.08:
        return _safe

    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = groq_key or openai_key
    use_groq = bool(groq_key)

    if not api_key:
        return _safe

    try:
        from openai import OpenAI
        if use_groq:
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            model = "llama-3.3-70b-versatile"
        else:
            client = OpenAI(api_key=api_key)
            model = "gpt-4o"
    except ImportError:
        return _safe

    # Compute per-band gap: output minus reference
    out_bands  = output_profile.get("band_energy_pct", {})
    ref_bands  = reference_profile.get("band_energy_pct", {})
    band_gaps  = {k: round(ref_bands.get(k, 0) - out_bands.get(k, 0), 4)
                  for k in ref_bands}  # positive = ref has more → boost output

    user_msg = f"""
Spectral distance between output and reference: {spectral_distance:.4f}
(0 = perfect match, 1 = completely different)

Per-band energy gap (reference minus output, positive = output needs more energy here):
{json.dumps(band_gaps, indent=2)}

Output spectral profile:
  centroid:    {output_profile.get('spectral_centroid_hz', '?')} Hz
  rolloff:     {output_profile.get('spectral_rolloff_hz', '?')} Hz
  crest_db:    {output_profile.get('crest_factor_db', '?')} dB
  dynamic_range: {output_profile.get('dynamic_range_db', '?')} dB

Reference spectral profile:
  centroid:    {reference_profile.get('spectral_centroid_hz', '?')} Hz
  rolloff:     {reference_profile.get('spectral_rolloff_hz', '?')} Hz
  crest_db:    {reference_profile.get('crest_factor_db', '?')} dB
  dynamic_range: {reference_profile.get('dynamic_range_db', '?')} dB

Please generate a small correction EQ (4–8 bands) to bring the output closer to the reference.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.15,
            max_tokens=512,
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
        # Clamp each band
        eq = []
        for band in raw.get("correction_eq", []):
            f = float(band.get("f", 1000))
            g = float(np.clip(band.get("gain_db", 0), -5.0, 5.0))
            q = float(np.clip(band.get("q", 1.0), 0.5, 2.0))
            if abs(g) >= 0.3 and 30 <= f <= 20000:
                eq.append({"f": f, "gain_db": g, "q": q})
        raw["correction_eq"] = eq[:8]
        logger.info(f"AI refinement: {len(eq)} correction bands — {raw.get('correction_summary', '')}")
        return raw
    except Exception as e:
        logger.warning(f"AI refinement EQ failed: {e}")
        return _safe


# ---------------------------------------------------------------------------
# AI Recipe Reviewer — used in the reference-matching pipeline
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM_PROMPT = """
You are a professional mixing engineer and audio DSP expert. Your PRIMARY GOAL is to make
the dry vocal sound as close as possible to the reference vocal. Apply effects AGGRESSIVELY
where needed — under-processing is just as bad as over-processing when the goal is reference matching.

You will be given:
  1. Raw signal-analysis results from a reference vocal track (effects detected by algorithms)
  2. Characteristics of the dry vocal that needs processing

Your job is to:
  - DEFAULT to applying every detected effect unless there is a very strong reason not to
  - Adjust parameters where the detected values need scaling up or down for the dry vocal
  - Explain briefly WHY you are keeping, adjusting, or (rarely) skipping each effect
  - Return a final clean recipe as strict JSON

Output format — ONLY valid JSON, no markdown, no extra text:
{
  "apply_eq": true | false,
  "apply_compression": true | false,
  "apply_reverb": true | false,
  "apply_saturation": true | false,
  "apply_width": true | false,
  "apply_chorus": true | false,
  "apply_delay": true | false,
  "apply_gate": true | false,
  "apply_tape": true | false,
  "apply_exciter": true | false,
  "apply_parallel_comp": true | false,
  "apply_autotune": true | false,
  "apply_vocal_layers": true | false,
  "adjustments": {
    "reverb_mix_scale": <float 0.5–1.5, default 1.0>,
    "eq_gain_scale": <float 0.5–1.5, default 1.0>,
    "chorus_mix_scale": <float 0.5–1.5, default 1.0>,
    "delay_mix": <float 0.1–0.4, default 0.25>,
    "harmony_strength_scale": <float 0.3–1.0, default 0.7>,
    "tape_mix_scale": <float 0.5–1.5, default 1.0>
  },
  "reasoning": {
    "eq": "<why keep/skip/adjust>",
    "compression": "<why>",
    "reverb": "<why>",
    "saturation": "<why>",
    "width": "<why>",
    "chorus": "<why>",
    "delay": "<why>",
    "gate": "<why>",
    "tape": "<why>",
    "exciter": "<why>",
    "parallel_comp": "<why>",
    "autotune": "<why>",
    "vocal_layers": "<why>"
  },
  "summary": "<1-2 sentence overall summary of what was done and why>"
}

The dry vocal stats include:
- rms, peak, crest_factor_db, dynamic_range_db  → dynamics / compression state of the raw vocal
- spectral_centroid_hz, spectral_rolloff_hz      → brightness of the raw vocal
- spectral_flatness, zero_crossing_rate          → noisiness / breathiness
- band_energy_pct                                → how much energy sits in sub/low/mid/high bands
- reverb_tail_ratio                              → whether the raw vocal already has reverb tails
- band_gap_vs_reference                          → per-band energy difference (ref minus dry);
                                                    positive = reference is brighter/fuller there,
                                                    negative = dry already has more energy there

Use these to SCALE parameters intelligently — not to skip effects:
- If dry reverb_tail_ratio > 0.2: scale reverb_mix_scale DOWN to 0.6–0.8, but still APPLY reverb
- If dry crest_factor_db < reference crest_factor_db: dry is already compressed → set lower ratio, still APPLY compression
- If dry spectral_centroid_hz is within 200Hz of reference: set eq_gain_scale to 0.7–0.9, but still APPLY EQ
- If band_gap_vs_reference shows large gaps (>3 dB): set eq_gain_scale to 1.1–1.3 to close those gaps
- If dry dynamic_range_db > 18dB: strong compression is needed
- For delay: if detected with any positive confidence value, APPLY it — use delay_mix 0.15–0.25 for subtle effect
- For chorus/vocal_layers: if the reference has ANY layering detected, APPLY it — scale chorus_mix_scale down if needed

SKIP rules (only these strict cases justify skipping):
- Skip gate ONLY if dry vocal has very clean background (spectral_flatness < 0.01 AND reverb_tail_ratio < 0.03)
- Skip reverb ONLY if dry reverb_tail_ratio > 0.35 (already heavily wet)
- Skip delay ONLY if detected delay confidence is literally 0 or null
- Skip vocal_layers ONLY if total_layers detected is 1 and n_doublers is 0
- Skip autotune ONLY if autotune strength is < 0.05 in detection

Default bias: if unsure, SET apply_* = true and use adjustments to scale the effect appropriately.
The goal is to make the dry vocal sound like the reference — lean toward applying everything detected.
""".strip()


def ai_review_recipe(
    detected_effects: dict,
    dry_vocal_stats: dict,
    reference_stats: dict,
) -> dict:
    """
    Have the LLM review the raw signal-analysis detections and produce an
    intelligent final recipe — deciding what to apply, what to skip, and
    what to adjust.

    Args:
        detected_effects: Dict of all effects detected from the reference
                          (eq, compression, reverb, chorus, delay, etc.)
        dry_vocal_stats:  Basic stats of the dry vocal (rms, lufs, spectral centroid, etc.)
        reference_stats:  Basic stats of the reference vocal (lufs, rms, etc.)

    Returns:
        Dict with keys: apply_*, adjustments, reasoning, summary
        Falls back to safe defaults if LLM call fails.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = groq_key or openai_key
    use_groq = bool(groq_key)

    _safe_defaults = {
        "apply_eq": True,
        "apply_compression": True,
        "apply_reverb": True,
        "apply_saturation": True,
        "apply_width": True,
        "apply_chorus": True,
        "apply_delay": True,
        "apply_gate": True,
        "apply_tape": True,
        "apply_exciter": True,
        "apply_parallel_comp": True,
        "apply_autotune": True,
        "apply_vocal_layers": True,
        "adjustments": {
            "reverb_mix_scale": 1.0,
            "eq_gain_scale": 1.0,
            "chorus_mix_scale": 1.0,
            "delay_mix": 0.25,
            "harmony_strength_scale": 0.7,
            "tape_mix_scale": 1.0,
        },
        "reasoning": {},
        "summary": "Signal detectors applied — AI review unavailable (no API key or call failed).",
    }

    if not api_key:
        logger.warning("No API key — skipping AI recipe review, using safe defaults")
        return _safe_defaults

    try:
        from openai import OpenAI
        if use_groq:
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            model = "llama-3.3-70b-versatile"
        else:
            client = OpenAI(api_key=api_key)
            model = "gpt-4o"
    except ImportError:
        return _safe_defaults

    user_message = f"""
Here are the raw signal-analysis results detected from the reference vocal:
{json.dumps(detected_effects, indent=2)}

Dry vocal characteristics (what we are STARTING WITH):
{json.dumps(dry_vocal_stats, indent=2)}

Reference vocal characteristics (what we are TRYING TO MATCH):
{json.dumps(reference_stats, indent=2)}

The "band_gap_vs_reference" in the dry vocal stats shows per-band energy difference
(reference minus dry). Positive = reference has MORE energy there → boost the dry vocal here.
Negative = dry already has MORE energy there → cut or leave it.

Your PRIMARY GOAL is to make the dry vocal sound like the reference. Apply all detected effects.
Use the dry vocal stats to SCALE parameters intelligently, not to skip effects.
Only skip an effect if one of the strict skip rules is met (see system prompt).
Return your final recipe as JSON with apply_* = true for everything that was detected.
""".strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1024,
        )

        raw_text = response.choices[0].message.content or "{}"
        logger.info("AI recipe review response received")
        result = json.loads(raw_text)

        # Ensure all required keys exist with safe fallbacks
        for key in _safe_defaults:
            if key not in result:
                result[key] = _safe_defaults[key]

        # Clamp adjustment scales to safe ranges
        adj = result.get("adjustments", {})
        adj["reverb_mix_scale"]       = float(max(0.3, min(1.5, adj.get("reverb_mix_scale", 1.0))))
        adj["eq_gain_scale"]          = float(max(0.3, min(1.5, adj.get("eq_gain_scale", 1.0))))
        adj["chorus_mix_scale"]       = float(max(0.3, min(1.5, adj.get("chorus_mix_scale", 1.0))))
        adj["delay_mix"]              = float(max(0.10, min(0.40, adj.get("delay_mix", 0.25))))
        adj["harmony_strength_scale"] = float(max(0.2, min(1.0,  adj.get("harmony_strength_scale", 0.7))))
        adj["tape_mix_scale"]         = float(max(0.3, min(1.5,  adj.get("tape_mix_scale", 1.0))))
        result["adjustments"] = adj

        return result

    except Exception as exc:
        logger.error(f"AI recipe review failed: {exc}")
        return _safe_defaults
