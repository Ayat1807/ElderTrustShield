"""
Resumable extraction of a FLAC audio shard.

Colab disconnects were traced to reading a large .tar sequentially from a
Drive-mounted (FUSE) path while creating tens of thousands of member files.
Fix: copy the archive to local disk first (removes Drive FUSE from the
per-file extraction path), and extract member-by-member so a disconnect
only costs progress since the last completed member, not the whole run.
See CLAUDE.md "Phase 2 kickoff findings" / Phase 1 "Current status" for
the full history of this issue.
"""
import shutil
import tarfile
from pathlib import Path


def extract_shard(
    drive_archive_path: Path,
    local_tar_path: Path,
    extract_dir: Path,
    expected_file_count: int,
    pattern: str = "*.flac",
) -> int:
    """Copy archive locally (if needed) and extract, resuming if interrupted.

    Returns the final file count. Raises ValueError if it doesn't match
    expected_file_count after extraction (safe to just re-run this function
    to resume).
    """
    if not local_tar_path.exists() or local_tar_path.stat().st_size != drive_archive_path.stat().st_size:
        print("Copying archive to local disk...")
        shutil.copy(drive_archive_path, local_tar_path)
    else:
        print("Local copy of archive already present.")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(local_tar_path, "r") as tf:
        members = tf.getmembers()
        total = len(members)
        print(f"Archive contains {total} members. Extracting (resumable)...")
        for i, member in enumerate(members):
            target = extract_dir / member.name
            if not target.exists():
                tf.extract(member, extract_dir, filter="data")
            if i % 2000 == 0:
                print(f"{i}/{total} processed", flush=True)

    final_count = sum(1 for _ in extract_dir.rglob(pattern))
    print("FINAL_FILE_COUNT=", final_count)
    if final_count != expected_file_count:
        raise ValueError(
            f"Extraction incomplete: expected {expected_file_count}, found {final_count}. "
            "Re-run this function — it resumes, skipping already-extracted files."
        )
    return final_count
