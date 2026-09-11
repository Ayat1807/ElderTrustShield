import sys
sys.path.insert(0, 'src')

import cv2
import gradio as gr
import torchaudio

from audio.predict import score_audio_file
from video.predict import score_video_file
from fusion import score_voice, score_video, combine_risks

MAX_AUDIO_SECONDS = 60
MAX_VIDEO_SECONDS = 30

TIER_COPY = {
    "voice": {
        "Low": "Nothing stood out — this is consistent with a genuine voice recording.",
        "Medium": "A few signals are worth a second look here, though nothing conclusive.",
        "High": "Several signs here are consistent with an AI-generated voice.",
    },
    "video": {
        "Low": "Nothing stood out — this is consistent with a genuine video.",
        "Medium": "A few signals are worth a second look here, though nothing conclusive.",
        "High": "Several signs here are consistent with an AI-generated or manipulated video.",
    },
    "combined": {
        "Low": "Taking both signals together, nothing here suggests this call is AI-generated.",
        "Medium": "Taking both signals together, there are some signs worth a closer look, though nothing conclusive yet.",
        "High": "Taking both signals together, this call shows meaningful signs of being AI-generated. It's worth confirming who you're speaking with another way.",
    },
}
TIER_COLORS = {"Low": "#6FC79A", "Medium": "#D8B15C", "High": "#E48576"}
TIER_BG = {"Low": "#16261F", "Medium": "#2A2718", "High": "#2B1B1A"}


def render_result_html(modality, tier, risk):
    color, bg = TIER_COLORS[tier], TIER_BG[tier]
    copy = TIER_COPY[modality][tier]
    pct = round(risk * 100)
    return f"""
    <div style="font-family: 'Work Sans', sans-serif;">
      <span style="display:inline-flex; align-items:center; gap:6px; padding:5px 13px;
                    border-radius:999px; font-size:13px; font-weight:600;
                    background:{bg}; color:{color};">
        <span style="width:7px;height:7px;border-radius:50%;background:{color};"></span>
        {tier} risk
      </span>
      <p style="font-size:14px; margin:10px 0 4px;">{copy}</p>
    <div style="font-size:12px; color:#9AA5AE;">{pct}% risk score</div>
    </div>
    """


def get_audio_duration(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform.shape[-1] / sample_rate


def get_video_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps else 0


def check_voice(audio_path, current_video_risk):
    if audio_path is None:
        return gr.update(value="<i>Choose an audio file first.</i>"), None, gr.update(visible=current_video_risk is not None)
    duration = get_audio_duration(audio_path)
    if duration > MAX_AUDIO_SECONDS:
        return gr.update(value=f"<i>This clip is {duration:.0f}s — longer than the {MAX_AUDIO_SECONDS}s limit for this prototype. Try a shorter clip.</i>"), None, gr.update(visible=current_video_risk is not None)
    try:
        risk = score_audio_file(audio_path)
    except Exception as e:
        return gr.update(value=f"<i>Could not analyze this file: {e}</i>"), None, gr.update(visible=False)
    result = score_voice(risk)
    return gr.update(value=render_result_html("voice", result["tier"], result["risk"])), risk, gr.update(visible=current_video_risk is not None)


def check_video(video_path, current_voice_risk):
    if video_path is None:
        return gr.update(value="<i>Choose a video file first.</i>"), None, gr.update(visible=current_voice_risk is not None)
    duration = get_video_duration(video_path)
    if duration > MAX_VIDEO_SECONDS:
        return gr.update(value=f"<i>This video is {duration:.0f}s — longer than the {MAX_VIDEO_SECONDS}s limit for this prototype. Try a shorter clip.</i>"), None, gr.update(visible=current_voice_risk is not None)
    try:
        risk = score_video_file(video_path)
    except ValueError as e:
        return gr.update(value=f"<i>{e} Try a clip with the face clearly visible.</i>"), None, gr.update(visible=False)
    except Exception as e:
        return gr.update(value=f"<i>Could not analyze this file: {e}</i>"), None, gr.update(visible=False)
    result = score_video(risk)
    return gr.update(value=render_result_html("video", result["tier"], result["risk"])), risk, gr.update(visible=current_voice_risk is not None)


def check_combined(voice_risk, video_risk):
    if voice_risk is None or video_risk is None:
        return gr.update(value="")
    result = combine_risks(voice_risk, video_risk)
    return render_result_html("combined", result["tier"], result["risk"])


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Work+Sans:wght@400;500;600&display=swap');
:root {
    --body-background-fill: #111823;
    --background-fill-primary: #161F2C;
    --block-background-fill: #161F2C;
    --block-border-color: #25303F;
    --block-label-background-fill: #1A2432;
    --block-label-text-color: #9AA5AE;
    --body-text-color: #E9ECEF;
    --body-text-color-subdued: #9AA5AE;
    --button-primary-background-fill: #C9A24A;
    --button-primary-background-fill-hover: #B08C3B;
    --button-primary-text-color: #111823;
}
.gradio-container { background: #111823 !important; color: #E9ECEF !important; font-family: 'Work Sans', sans-serif !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { font-family: 'Fraunces', serif !important; color: #E9ECEF !important; }
.gradio-container label, .gradio-container .block { color: #E9ECEF !important; }
.gradio-container .block { background: #161F2C !important; border-color: #25303F !important; }
button.primary { background: #C9A24A !important; border-color: #C9A24A !important; color: #111823 !important; }
button.primary:hover { background: #B08C3B !important; }
"""

with gr.Blocks(css=CUSTOM_CSS, title="Signal Check") as demo:
    gr.HTML("""
      <div style="text-align:center; margin-bottom:8px;">
        <h1 style="font-weight:600; color:#E9ECEF;">A calm second opinion on suspicious calls</h1>
        <p style="color:#9AA5AE;">Upload a voice clip or a video recording to check for signs it may be AI-generated.</p>
      </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Voice")
            audio_in = gr.Audio(type="filepath", label="Voice clip")
            voice_btn = gr.Button("Check this voice", variant="primary")
            voice_result = gr.HTML()
        with gr.Column():
            gr.Markdown("### Video")
            video_in = gr.Video(label="Video recording")
            video_btn = gr.Button("Check this video", variant="primary")
            video_result = gr.HTML()

    combined_btn = gr.Button("See the combined picture", visible=False)
    combined_result = gr.HTML()

    voice_risk_state = gr.State(None)
    video_risk_state = gr.State(None)

    voice_btn.click(check_voice, inputs=[audio_in, video_risk_state], outputs=[voice_result, voice_risk_state, combined_btn])
    video_btn.click(check_video, inputs=[video_in, voice_risk_state], outputs=[video_result, video_risk_state, combined_btn])
    combined_btn.click(check_combined, inputs=[voice_risk_state, video_risk_state], outputs=[combined_result])

    gr.HTML("""
            <div style="text-align:center; margin-top:24px; padding-top:16px; border-top:1px solid #25303F;">
                <p style="font-size:12.5px; color:#9AA5AE;">This is a risk signal meant to support your own judgment — not a verdict, and not a guarantee.
        Uploaded files are not stored or retained. If something feels off, it's always reasonable to verify who you're speaking with another way.</p>
      </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
