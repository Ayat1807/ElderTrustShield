"""
Build a labels CSV by matching the ASVspoof5 protocol file against
extracted audio files.

Column parsing is CONFIRMED (not guessed) against a published description
of the ASVspoof5 protocol schema: space-separated columns —
speaker_id, filename, gender, [reserved fields], attack_configuration
(AC1/AC2/AC3), attack_algorithm (A##), label (bonafide/spoof), trailing
placeholder field. So parts[1]=filename, parts[-2]=label,
parts[-3]=attack_algorithm. See CLAUDE.md for the cross-check history.
"""
import csv
from pathlib import Path


def build_labels_csv(protocol_path: Path, extract_dir: Path, out_csv: Path, pattern: str = "*.flac") -> dict:
    """Match extracted audio files against the protocol file and write a labels CSV.

    Returns a dict of {'bonafide': n, 'spoof': n} counts.
    """
    present = {p.stem: p.relative_to(extract_dir).as_posix() for p in sorted(extract_dir.rglob(pattern))}

    rows = []
    with protocol_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            filename_key, label, attack_algorithm = parts[1], parts[-2], parts[-3]
            if label in {"bonafide", "spoof"} and filename_key in present:
                rows.append((present[filename_key], label, attack_algorithm))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label", "attack_algorithm"])
        writer.writerows(rows)

    counts = {"bonafide": 0, "spoof": 0}
    for _, label, _ in rows:
        counts[label] += 1

    print("TOTAL_MATCHED_ROWS=", len(rows))
    print("REAL_BONAFIDE=", counts["bonafide"])
    print("FAKE_SPOOF=", counts["spoof"])
    print("CSV_PATH=", out_csv)
    return counts
