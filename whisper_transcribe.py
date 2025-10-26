#!/usr/bin/env python3
"""
whisper_transcribe.py

Verwendung:
    ./run-transcriber.sh /pfad/zur/aufnahme.m4a [--model-size small|medium|large-v3] [--device auto|cpu|cuda]

Funktion:
- Dekodiert beliebiges Audio (m4a/mp3/wav/...) per ffmpeg zu Mono-PCM 16kHz Float32.
- Nutzt ein mehrsprachiges Whisper-Modell (Standard: whisper-small) für
  deutsche Spracherkennung -> deutscher Text (keine Übersetzung ins Englische).
- Schreibt Ergebnis als <eingabe>.txt.
- Führt ein simples Postprocessing durch: nach jedem Punkt "." werden zwei
  Newlines eingefügt, um Absätze sichtbarer zu machen.

Hinweise:
- ffmpeg muss im PATH sein.
- CPU-Modus ist langsam aber stabil. GPU-Modus ist schnell, kann aber bei
  großen Modellen VRAM-Fehler (Out Of Memory) werfen.
- Bei OOM auf GPU wird automatisch auf CPU zurückgefallen.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.base import PipelineException

###############################################################################
# Audio-Dekodierung via ffmpeg
###############################################################################


def ffmpeg_decode_to_mono_16k_float32(audio_path: Path) -> tuple[np.ndarray, int]:
    """
    Wandelt Eingabe-Audio nach Mono 16 kHz Float32 PCM um (per ffmpeg) und
    liefert (waveform, sample_rate) zurück.

    So umgehen wir libsndfile/soundfile-Probleme bei .m4a.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",  # nur Fehler ausgeben
        "-i",
        str(audio_path),  # Eingabedatei
        "-f",
        "f32le",  # rohes 32-bit float little-endian
        "-acodec",
        "pcm_f32le",  # zwinge PCM Float32
        "-ac",
        "1",  # Mono
        "-ar",
        "16000",  # 16 kHz
        "pipe:1",  # schreibe rohe Samples nach stdout
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg nicht gefunden. Bitte ffmpeg installieren (z.B. apt install ffmpeg)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "ffmpeg konnte die Audiodatei nicht dekodieren:\n"
            + e.stderr.decode("utf-8", errors="ignore")
        ) from e

    audio_bytes = proc.stdout
    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
    sample_rate = 16000
    return audio_np, sample_rate


###############################################################################
# Modell-/Pipeline-Helfer
###############################################################################


def resolve_model_id(model_size: str) -> str:
    """
    Mappt --model-size auf einen Whisper-Checkpoint von Hugging Face.

    Alle hier genannten Modelle sind mehrsprachig (Deutsch-fähig):
    - openai/whisper-small
    - openai/whisper-medium
    - openai/whisper-large-v3
    usw.
    """
    size = model_size.lower()
    if size in ("large-v3", "large", "largev3"):
        return "openai/whisper-large-v3"
    if size in ("medium", "med"):
        return "openai/whisper-medium"
    if size in ("small", "sm"):
        return "openai/whisper-small"
    if size in ("base",):
        return "openai/whisper-base"
    if size in ("tiny",):
        return "openai/whisper-tiny"
    # Fallback
    return "openai/whisper-small"


def pick_device(device_arg: str) -> str:
    """
    Wählt Device-String für die Pipeline:
    - "auto": nimm CUDA wenn verfügbar, sonst CPU
    - "cuda": zwinge CUDA
    - "cpu": zwinge CPU
    """
    dev = device_arg.lower()
    if dev == "cpu":
        return "cpu"
    if dev in ("cuda", "gpu"):
        return "cuda:0"
    # auto
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def build_asr_pipeline(model_id: str, device: str, language_code: str = "de"):
    """
    Erzeugt die Hugging Face ASR-Pipeline mit Whisper.

    - dtype:
        * auf GPU float16 (spart VRAM)
        * auf CPU float32
    - generate_kwargs:
        "language": "de", "task": "transcribe"
        => deutsche Sprache rein, deutscher Text raus
        => keine automatische Übersetzung ins Englische
    - chunk_length_s:
        wir schneiden Audio in ~20s Blöcke; das hält Speicherbedarf klein,
        kann aber an Chunk-Grenzen leichte Wiederholungen / Auslassungen bringen.
    - ignore_warning=True:
        unterdrückt die laute Hinweis-Warnung der Pipeline.
    """
    if device.startswith("cuda"):
        model_dtype = torch.float16
        batch_size = 8  # kleiner halten für weniger VRAM
        chunk_length_s = 20
    else:
        model_dtype = torch.float32
        batch_size = 1  # CPU braucht kleine Batches
        chunk_length_s = 20

    # Modell laden
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    generate_kwargs = {
        "language": language_code,  # "de"
        "task": "transcribe",  # gleiche Sprache behalten
    }

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        dtype=model_dtype,
        device=device,
        ignore_warning=True,
        generate_kwargs=generate_kwargs,
    )

    return asr_pipe


###############################################################################
# Transkription + Postprocessing
###############################################################################


def transcribe_waveform(asr_pipe, audio_np: np.ndarray, sample_rate: int) -> str:
    """
    Übergibt rohe PCM-Daten + Samplerate direkt an die Pipeline
    und gibt den Roh-Transkript-Text zurück.
    """
    result = asr_pipe(
        {
            "array": audio_np,
            "sampling_rate": sample_rate,
        }
    )
    return result["text"]


def postprocess_text(text: str) -> str:
    """
    Füge nach jedem Punkt '.' zwei Newlines ein.

    Wichtig:
    - stumpf und brutal: jedes '.' wird zu '.\\n\\n'
    - keine Spezialbehandlung für Abkürzungen ('z.B.') oder Zahlen ('3.14')
    - falls du später was Schlaueres willst (Regex für Satzende usw.),
      können wir das feinjustieren.
    """
    return text.replace(".", ".\n\n")


def run_transcription(
    audio_path: Path,
    out_path: Path,
    model_size: str,
    device_pref: str,
):
    """
    Orchestriert:
    - Audio dekodieren
    - Pipeline bauen (GPU oder CPU)
    - bei CUDA OOM automatisch auf CPU zurückfallen
    - Postprocessing anwenden
    - Ergebnis schreiben
    """
    # 1. Audio dekodieren
    audio_np, sr = ffmpeg_decode_to_mono_16k_float32(audio_path)

    # 2. Modell & Device auswählen
    model_id = resolve_model_id(model_size)
    device = pick_device(device_pref)

    # 3. Versuchen, auf gewünschtem Device zu laufen
    try:
        asr_pipe = build_asr_pipeline(model_id, device)
        raw_text = transcribe_waveform(asr_pipe, audio_np, sr)
    except torch.cuda.OutOfMemoryError:
        print("WARN: GPU VRAM reicht nicht, wechsle auf CPU ...", flush=True)
        asr_pipe = build_asr_pipeline(model_id, "cpu")
        raw_text = transcribe_waveform(asr_pipe, audio_np, sr)
    except PipelineException as e:
        if "CUDA out of memory" in str(e) and device.startswith("cuda"):
            print("WARN: GPU VRAM reicht nicht, wechsle auf CPU ...", flush=True)
            asr_pipe = build_asr_pipeline(model_id, "cpu")
            raw_text = transcribe_waveform(asr_pipe, audio_np, sr)
        else:
            raise

    # 4. Postprocessing: nach jedem Punkt zwei Zeilenumbrüche
    final_text = postprocess_text(raw_text)

    # 5. Ergebnis speichern
    out_path.write_text(final_text, encoding="utf-8")
    print(f"Transkription abgeschlossen:\n  {out_path}")


###############################################################################
# main
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transkribiert gesprochene deutsche Sprache -> deutscher Text "
            "(mehrsprachige Whisper-Modelle), inkl. m4a-Support.\n"
            "Fügt nach jedem Punkt einen doppelten Zeilenumbruch ein."
        )
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Pfad zur Audiodatei (z.B. aufnahme.m4a)",
    )

    parser.add_argument(
        "--model-size",
        default="small",
        help=(
            "Whisper-Modellgröße: tiny | base | small | medium | large-v3 "
            "(default: small). Größer = genauer, aber mehr RAM/VRAM."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "auto | cpu | cuda  (default: auto)\n"
            "auto = benutze CUDA falls verfügbar, sonst CPU\n"
            "cpu  = zwinge CPU (stabil, langsamer)\n"
            "cuda = zwinge GPU (schnell, kann OOM liefern)"
        ),
    )

    args = parser.parse_args()

    in_path = args.input_file
    if not in_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {in_path}")

    out_path = in_path.with_suffix(".txt")

    run_transcription(
        audio_path=in_path,
        out_path=out_path,
        model_size=args.model_size,
        device_pref=args.device,
    )


if __name__ == "__main__":
    main()
