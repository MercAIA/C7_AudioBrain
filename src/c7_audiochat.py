import os
import numpy as np
import librosa
import soundfile as sf
from openai import OpenAI

# =================================================================
# Clients
# =================================================================

# اگر در آینده خواستی OpenAI را فعال کنی، این client هست
oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))

# Groq برای ASR (Whisper)
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# =================================================================
# Audio Feature Extraction
# =================================================================

def extract_features(wav_path: str) -> dict:
    wav, sr = librosa.load(wav_path, sr=None)

    energy = np.mean(wav ** 2)
    pitch = librosa.yin(wav, fmin=80, fmax=400, sr=sr)
    pitch_mean = float(np.mean(pitch)) if pitch.size > 0 else 0.0

    rate = len(wav) / sr

    features = {
        "energy": float(energy),
        "pitch": float(pitch_mean),
        "rate": float(rate),
    }
    return features


# =================================================================
# AudioState (C7 Listener State)
# =================================================================

def build_audio_state(curr: dict, baseline: dict = None) -> dict:
    if baseline is None:
        baseline = curr

    delta_pitch = curr["pitch"] - baseline["pitch"]
    delta_energy = curr["energy"] - baseline["energy"]
    delta_rate = curr["rate"] - baseline["rate"]

    tension = max(0.0, min(1.0, abs(delta_pitch) * 2.0))
    stability = 1.0 - tension
    warmth = max(0.0, min(1.0, 1.0 - abs(delta_energy)))
    novelty = max(0.0, min(1.0, abs(delta_rate)))

    return {
        "tension": tension,
        "stability": stability,
        "warmth": warmth,
        "novelty": novelty,
        "delta_pitch": delta_pitch,
        "delta_energy": delta_energy,
        "delta_rate": delta_rate,
    }


# =================================================================
# ASR via Groq Whisper-large-v3
# =================================================================

def asr_transcribe(wav_path: str) -> str:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"فایل صوتی پیدا نشد: {wav_path}")

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY تنظیم نشده.")

    print("\nدر حال انجام ASR با Groq (whisper-large-v3) ...")

    with open(wav_path, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="text",
            temperature=0.0,
        )

    if isinstance(resp, str):
        return resp.strip()
    if hasattr(resp, "text"):
        return resp.text.strip()

    return str(resp).strip()


# =================================================================
# Heart / Intent / Decision
# =================================================================

def heart_plan(audio_state: dict) -> dict:
    tension = audio_state["tension"]
    warmth = audio_state["warmth"]

    if tension > 0.5:
        intent = "soothe"
    elif warmth > 0.5:
        intent = "empathic"
    else:
        intent = "simple_ack"

    plan = {
        "intent": intent,
        "tone": "calm" if intent == "soothe" else "neutral"
    }
    return plan


# =================================================================
# Awareness (Meta Feedback)
# =================================================================

def awareness_feedback(text_in: str, plan: dict) -> str:
    return f"(Awareness) intent={plan['intent']}  |  text='{text_in}'"


# =================================================================
# LLM (Stub) — بدون نیاز به OpenAI Billing
# =================================================================

def llm_generate_reply(text_in: str, audio_state: dict, plan: dict) -> str:
    intent = plan.get("intent", "neutral")

    prefix = ""
    if intent == "soothe":
        prefix = "می‌فهمم انرژی صدات یه‌کم بالاست، آرام جواب می‌دم. "
    elif intent == "empathic":
        prefix = "حس می‌کنم پشت حرف‌هات احساس هست. "
    elif intent == "simple_ack":
        prefix = "شنیدم چی گفتی. "

    return prefix + f"(نسخه موقت C7) گفتی: «{text_in}»"


# =================================================================
# TTS (Stub)
# =================================================================

def tts_synthesize(text: str, tone: str, out_path: str):
    txt_path = out_path.replace(".wav", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"[tone={tone}]\n{text}\n")
    print(f"\n(نسخه موقت) خروجی صوت به‌صورت متن ذخیره شد: {txt_path}")


# =================================================================
# Main Pipeline
# =================================================================

def process_voice(uid: str, wav_path: str, out_path: str):
    print(f"\n--- Processing for User {uid} ---")

    feats = extract_features(wav_path)
    baseline = feats
    audio_state = build_audio_state(feats, baseline)

    print("\n--- AudioState ---")
    for k, v in audio_state.items():
        print(f"{k}: {v:.3f}")

    text_in = asr_transcribe(wav_path)
    print(f"\nمتن ورودی (ASR): {text_in}")

    plan = heart_plan(audio_state)
    print(f"\nPlan: {plan}")

    fb = awareness_feedback(text_in, plan)
    print(f"Feedback: {fb}")

    reply_text = llm_generate_reply(text_in, audio_state, plan)
    print(f"\nFinal Reply: {reply_text}")

    tts_synthesize(reply_text, plan["tone"], out_path)


# =================================================================
# Entry
# =================================================================

if __name__ == "__main__":
    print("مسیر فایل ویس ورودی را بده: ", end="")
    wav_path = input().strip()

    uid = "u1"
    out_path = "c7_reply.wav"

    process_voice(uid, wav_path, out_path)