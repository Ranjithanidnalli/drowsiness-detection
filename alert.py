"""
alert.py — Escalating audio alerts + text-to-speech.

Levels:
  0 = Alert    → silence
  1 = Mild     → periodic single beep
  2 = Drowsy   → sustained repeating buzzer + TTS voice warning

The buzzer runs on a background thread so it never freezes the video loop.
Audio backend: pygame tones (preferred) with a Windows winsound fallback.
"""

import threading
import time

import numpy as np

# ── pygame mixer (preferred tone backend) ─────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    _PYGAME_OK = True
except Exception:
    _PYGAME_OK = False

# ── winsound (guaranteed Windows fallback) ────────────────────────────────────
try:
    import winsound
    _WINSOUND_OK = True
except Exception:
    _WINSOUND_OK = False

# ── pyttsx3 text-to-speech ────────────────────────────────────────────────────
try:
    import pyttsx3
    _TTS_ENGINE = pyttsx3.init()
    _TTS_ENGINE.setProperty("rate", 160)
    _TTS_ENGINE.setProperty("volume", 1.0)
    _TTS_OK = True
except Exception:
    _TTS_OK = False


def _generate_beep(frequency: int = 880, duration_ms: int = 400, volume: float = 0.8):
    """Generate a sine-wave beep sized to the mixer's actual channel count."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    mono = (volume * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    # SDL often forces stereo even when channels=1 was requested. Shape the
    # array to match the REAL channel count, otherwise make_sound() raises
    # "Array must be 2-dimensional for stereo mixer".
    init = pygame.mixer.get_init()
    channels = init[2] if init else 1
    wave = np.column_stack([mono] * channels) if channels >= 2 else mono
    return pygame.sndarray.make_sound(np.ascontiguousarray(wave))


_mild_sound = None
_drowsy_sound = None

if _PYGAME_OK:
    try:
        _mild_sound   = _generate_beep(frequency=660, duration_ms=300, volume=0.6)
        _drowsy_sound = _generate_beep(frequency=900, duration_ms=450, volume=1.0)
    except Exception:
        _PYGAME_OK = False


_tts_lock = threading.Lock()
_tts_busy = False


def _speak(text: str) -> None:
    global _tts_busy
    if not _TTS_OK:
        return
    with _tts_lock:
        if _tts_busy:
            return
        _tts_busy = True
    try:
        _TTS_ENGINE.say(text)
        _TTS_ENGINE.runAndWait()
    except Exception:
        pass
    finally:
        with _tts_lock:
            _tts_busy = False


class AlertManager:
    """
    Call update(level, reason) each frame.  level: 0=Alert, 1=Mild, 2=Drowsy.

    A background thread owns all sound output:
      • level 2 → continuous urgent buzzer + spoken warning every few seconds
      • level 1 → a single beep roughly every MILD_INTERVAL seconds
      • level 0 → silence
    """

    MILD_INTERVAL = 2.5    # seconds between mild beeps
    TTS_INTERVAL  = 4.0    # seconds between spoken drowsy warnings

    def __init__(self):
        self._level = 0
        self._reason = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_mild = 0.0
        self._last_tts = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, level: int, reason: str = "") -> None:
        """Record the latest level/reason; the alarm thread reacts to it."""
        with self._lock:
            self._level = level
            self._reason = reason

    # ── Background alarm loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                level = self._level
                reason = self._reason

            if level >= 2:
                self._buzz(drowsy=True)
                now = time.time()
                if now - self._last_tts >= self.TTS_INTERVAL:
                    self._last_tts = now
                    msg = reason if reason else "Wake up! Please pull over and rest."
                    threading.Thread(target=_speak, args=(msg,), daemon=True).start()

            elif level == 1:
                now = time.time()
                if now - self._last_mild >= self.MILD_INTERVAL:
                    self._last_mild = now
                    self._buzz(drowsy=False)
                else:
                    self._stop.wait(0.05)
            else:
                self._stop.wait(0.05)

    def _buzz(self, drowsy: bool) -> None:
        """Play one buzzer cycle (blocking on THIS thread only)."""
        if _PYGAME_OK:
            sound = _drowsy_sound if drowsy else _mild_sound
            if sound is not None:
                sound.play()
                # Pace to the clip length so drowsy buzzes run back-to-back.
                self._stop.wait(sound.get_length() + (0.05 if drowsy else 0.0))
                return
        if _WINSOUND_OK:
            freq = 900 if drowsy else 660
            dur  = 450 if drowsy else 300
            try:
                winsound.Beep(freq, dur)   # blocking, but on the alarm thread
            except Exception:
                self._stop.wait(0.3)

    def stop(self) -> None:
        self._stop.set()
        if _PYGAME_OK:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
