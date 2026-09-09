def classify_risk_tier(risk_score: float) -> str:
    """
    risk_score: 0-1 probability of being AI-generated/fake.
    High threshold set by explicit product decision: >75%.
    Medium/Low boundary at 30% - carried over from the original proposal.
    """
    if risk_score > 0.75:
        return "High"
    elif risk_score > 0.30:
        return "Medium"
    else:
        return "Low"


def score_voice(voice_risk: float) -> dict:
    """Shown immediately once voice inference completes - independent of video."""
    return {"modality": "voice", "risk": voice_risk, "tier": classify_risk_tier(voice_risk)}


def score_video(video_risk: float) -> dict:
    """Shown immediately once video inference completes - independent of voice."""
    return {"modality": "video", "risk": video_risk, "tier": classify_risk_tier(video_risk)}


def combine_risks(voice_risk: float, video_risk: float) -> dict:
    """
    Only called when the user explicitly asks to see the combined view -
    both individual results should already be visible on their own before
    this is ever invoked. Noisy-OR: answers "what's the chance at least
    one channel is fake" rather than averaging, so a strong single-channel
    signal doesn't get diluted by a clean one.
    """
    combined = 1 - (1 - voice_risk) * (1 - video_risk)
    return {"modality": "combined", "risk": combined, "tier": classify_risk_tier(combined)}
