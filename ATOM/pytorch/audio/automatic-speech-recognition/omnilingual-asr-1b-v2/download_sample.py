"""Fetch one CC BY 4.0 LibriSpeech utterance, attribution and reference transcript."""

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

from artifact import digest

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
UTTERANCE = "6930-75918-0000"
PREFIX = "LibriSpeech/test-clean/6930/75918/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/sample"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        f"{PREFIX}{UTTERANCE}.flac": "sample.flac",
        f"{PREFIX}6930-75918.trans.txt": "transcripts.txt",
        "LibriSpeech/LICENSE.TXT": "LICENSE.TXT",
        "LibriSpeech/README.TXT": "README.TXT",
    }
    # Read only as far as the requested members. Never unpack arbitrary archive paths.
    with (
        urllib.request.urlopen(URL) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            if member.name in wanted:
                (args.output_dir / wanted.pop(member.name)).write_bytes(
                    archive.extractfile(member).read()
                )
            if not wanted:
                break
    if wanted:
        raise FileNotFoundError(
            f"missing expected LibriSpeech members: {sorted(wanted)}"
        )
    lines = (args.output_dir / "transcripts.txt").read_text().splitlines()
    reference = next(
        line.split(" ", 1)[1] for line in lines if line.startswith(UTTERANCE + " ")
    )
    metadata = {
        "utterance": UTTERANCE,
        "url": URL,
        "license": "CC BY 4.0",
        "dataset": "LibriSpeech, Vassil Panayotov et al.",
        "source_page": "https://www.openslr.org/12",
        "audio_sha256": digest(args.output_dir / "sample.flac"),
        "reference": reference,
    }
    (args.output_dir / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
