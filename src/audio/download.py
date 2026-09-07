"""
Download the ASVspoof5 protocol file and a FLAC audio shard.

Large downloads (roughly >1GB) must go through Colab, not a local machine —
confirmed 2026-09-03: a 6.65GB file stalled completely on the local network
(with or without HF's Xet backend), while the same file downloaded via
Colab in under 2 minutes at ~69MB/s. See CLAUDE.md for details.
"""
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download  # noqa: E402


def download_protocols(local_base: Path) -> Path:
    """Download and return the path to the extractable protocol tar."""
    local_base.mkdir(parents=True, exist_ok=True)
    protocol_tar = local_base / "ASVspoof5_protocols.tar"
    if protocol_tar.exists():
        print("Protocol archive already exists:", protocol_tar)
        return protocol_tar
    print("Downloading protocol archive...")
    return Path(
        hf_hub_download(
            repo_id="jungjee/asvspoof5",
            repo_type="dataset",
            filename="ASVspoof5_protocols.tar",
            local_dir=str(local_base),
        )
    )


def download_audio_shard(drive_base: Path, shard_filename: str = "flac_D_aa.tar") -> Path:
    """Download an audio shard to Drive (persisted; re-download is cheap if lost)."""
    drive_base.mkdir(parents=True, exist_ok=True)
    archive_path = drive_base / shard_filename
    if archive_path.exists():
        print("Audio archive already exists on Drive:", archive_path)
        return archive_path
    print(f"Downloading {shard_filename} to Drive...")
    return Path(
        hf_hub_download(
            repo_id="jungjee/asvspoof5",
            repo_type="dataset",
            filename=shard_filename,
            local_dir=str(drive_base),
        )
    )
