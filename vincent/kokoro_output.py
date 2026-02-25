"""Kokoro text-to-speech output helpers.

Wraps Kokoro initialization and audio playback for assistant responses.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import sounddevice as sd


class KokoroSpeaker:  # pylint: disable=too-few-public-methods
    """Generate and play speech audio from assistant text with Kokoro."""

    def __init__(
        self,
        lang_code: str,
        voice: str,
        speed: float,
    ) -> None:
        """Initialize Kokoro pipeline and playback parameters."""
        warnings.filterwarnings(
            "ignore",
            message=("dropout option adds dropout after all but last recurrent layer"),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                "`torch.nn.utils.weight_norm` is deprecated in favor of "
                "`torch.nn.utils.parametrizations.weight_norm`"
            ),
            category=FutureWarning,
        )

        try:
            kokoro_module = importlib.import_module("kokoro")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = (
                "Kokoro could not be imported. Install with `uv add kokoro` and "
                "use Python 3.10-3.13 for voice mode."
            )
            raise RuntimeError(msg) from exc

        kpipeline = getattr(kokoro_module, "KPipeline", None)
        if kpipeline is None:
            msg = "Installed kokoro package does not expose KPipeline"
            raise RuntimeError(msg)

        try:
            self._pipeline = kpipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = (
                "Kokoro failed to initialize. This is often a Python-version "
                "compatibility problem (Kokoro stack currently targets Python "
                "3.10-3.13)."
            )
            raise RuntimeError(msg) from exc
        self._voice = voice
        self._speed = speed
        self._sample_rate = 24000

    def speak(self, text: str) -> None:
        """Convert text to speech and play it through the default audio output."""
        generator = self._pipeline(
            text,
            voice=self._voice,
            speed=self._speed,
            split_pattern=r"\n+",
        )

        chunks = [audio for _, _, audio in generator if len(audio)]
        if not chunks:
            return

        output_audio = np.concatenate(chunks)
        try:
            sd.play(output_audio, samplerate=self._sample_rate)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            raise
