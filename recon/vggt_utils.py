"""Shared VGGT loading."""

from __future__ import annotations

import sys

MODEL_ID = "facebook/VGGT-1B"


def load_vggt():
    """VGGT-1B from the local Hugging Face cache ($HF_HOME), downloading only if it is not there.

    from_pretrained() otherwise asks the Hub for updates on every call (the "unauthenticated
    requests" warning); with local_files_only it never touches the network once cached.
    """
    from vggt.models.vggt import VGGT

    try:
        return VGGT.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:  # not cached yet (first run) -> download once (~5 GB to $HF_HOME)
        print(f"{MODEL_ID} not in the local cache, downloading ...", file=sys.stderr)
        return VGGT.from_pretrained(MODEL_ID)
