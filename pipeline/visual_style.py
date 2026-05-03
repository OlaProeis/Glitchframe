"""Reactive shader UX bundle: example SDXL prompts, palettes, typography, motion.

The Gradio **Visual style** tab drives these via shader selection instead of YAML
presets. :func:`style_preset_id` feeds background cache manifests.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.builtin_shaders import BUILTIN_SHADERS


@dataclass(frozen=True)
class ShaderVisualBundle:
    example_prompt: str
    typo_style: str
    colors: tuple[str, ...]
    motion_flavor: str


NONE_SHADER = "none"

# Canonical order shown in UI (excluding none, which is listed first separately).
PRIMARY_REACTIVE_SHADER_ORDER: tuple[str, ...] = (
    NONE_SHADER,
    "void_ascii_bg",
    "spectral_milkdrop",
    "tunnel_flight",
    "synth_grid",
)

_SHADER_BUNDLES: dict[str, ShaderVisualBundle] = {
    NONE_SHADER: ShaderVisualBundle(
        example_prompt=(
            "Cinematic ultra-wide music video plate, striking composition, "
            "crisp focal subject, saturated practical lighting, high contrast, "
            "no atmospheric haze, master-quality detail"
        ),
        typo_style="beat-shake",
        colors=(
            "#0D0221",
            "#240046",
            "#7B2CBF",
            "#FF6B35",
            "#FFF3B0",
        ),
        motion_flavor=(
            "slow cinematic motion, subtle parallax, smooth continuous framing, "
            "stable camera"
        ),
    ),
    "void_ascii_bg": ShaderVisualBundle(
        example_prompt=(
            "Deep void negative space, soft digital grain, no literal scenery — "
            "abstract darkness with subtle colour haze only, minimal, high contrast, 4K clean"
        ),
        typo_style="pop-in",
        colors=(
            "#0A0A12",
            "#6B5B95",
            "#E8E6F0",
            "#FF3B5C",
            "#7DF9FF",
        ),
        motion_flavor=(
            "very slow push-in, calm deliberate framing, subtle light shifts, "
            "meditative pacing"
        ),
    ),
    "spectral_milkdrop": ShaderVisualBundle(
        example_prompt=(
            "Volumetric holographic light sculpture in a dark void, infinite chrome "
            "helixes and cyan-magenta caustics, lens accents on transitions, "
            "bar-synced colour sweeps, ultra-clean futuristic concert visuals, sharp abstract"
        ),
        typo_style="beat-shake",
        colors=(
            "#070018",
            "#00E5FF",
            "#B026FF",
            "#FF2D95",
            "#FFF5B7",
        ),
        motion_flavor=(
            "slow orbital drift, layered parallax, shimmering energy fields, "
            "ethereal floating motion"
        ),
    ),
    "tunnel_flight": ShaderVisualBundle(
        example_prompt=(
            "Dark cyberpunk void, infinite wireframe tunnel vanishing into "
            "red-orange emissive lattice, deep black negative space, "
            "cool blue-violet depth, high-contrast neon, first-person flight, cinematic wide"
        ),
        typo_style="beat-shake",
        colors=(
            "#FF2A1A",
            "#FF6B2C",
            "#6B4DFF",
            "#9D4EDD",
            "#FF1B8D",
        ),
        motion_flavor=(
            "smooth forward drift through the tunnel, gentle horizon sway, "
            "cinematic depth cues, coherent parallax"
        ),
    ),
    "synth_grid": ShaderVisualBundle(
        example_prompt=(
            "Neon 1980s retrowave skyline, chrome perspective grid, palm silhouettes, "
            "magenta and cyan rim light, subtle film grain, cinematic wide shot, ultra sharp"
        ),
        typo_style="beat-shake",
        colors=(
            "#1A0A2E",
            "#16213E",
            "#00F5FF",
            "#FF2EE6",
            "#E8F7FF",
        ),
        motion_flavor=(
            "slow forward drive, gentle horizon sway, distant city parallax, "
            "cinematic retrowave motion"
        ),
    ),
}


def canonical_reactive_shader_stem(raw: str | None) -> str:
    """Return allowlisted reactive shader stem or :data:`NONE_SHADER`."""
    s = str(raw or "").strip()
    if s in BUILTIN_SHADERS:
        return s
    return NONE_SHADER


def shader_style_bundle(stem: str) -> ShaderVisualBundle:
    """Defaults for typography, palette, prompts, and AnimateDiff motion."""
    key = canonical_reactive_shader_stem(stem)
    return _SHADER_BUNDLES[key]


def style_preset_id(shader_stem: str) -> str:
    """Cache / manifest preset key for backgrounds (e.g. ``style-synth_grid``)."""
    return f"style-{canonical_reactive_shader_stem(shader_stem)}"


def motion_flavor_for_style_preset(preset_id: str) -> str | None:
    """Resolve AnimateDiff motion snippet for ``style-<shader>`` ids; else ``None``."""
    pid = str(preset_id).strip()
    prefix = "style-"
    if not pid.startswith(prefix):
        return None
    stem = canonical_reactive_shader_stem(pid[len(prefix) :])
    return _SHADER_BUNDLES[stem].motion_flavor.strip()
