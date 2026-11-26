"""MOSS-TTSD dialogue text, decided before any model loads.

``normalize_text``, ``merge_consecutive_speaker_tags``, ``normalize_prompt_text`` and
``validate_dialogue_text`` are the creator's functions from MOSS-TTS
``clis/moss_ttsd_app.py`` at 934d6826b084c46a0d033402174d5f8ac4ed2519 (Apache-2.0),
kept as written; ``speakers`` maps the job's ordered reference slots onto them.
"""

from __future__ import annotations

import re

MIN_SPEAKERS = 1
MAX_SPEAKERS = 5


def normalize_text(text: str) -> str:
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)
    remove_chars = "【】《》（）『』「」" '"-_“”～~‘’'

    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = matched.groups() if matched else ("", seg)

        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)

        content = content.replace("——", "，")
        content = content.replace("……", "，")
        content = content.replace("...", "，")
        content = content.replace("⸺", "，")
        content = content.replace("―", "，")
        content = content.replace("—", "，")
        content = content.replace("…", "，")

        internal_punct_map = str.maketrans(
            {"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}
        )
        content = content.translate(internal_punct_map)
        content = content.strip()
        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)

        if len(content) > 1:
            last_ch = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            body = content[:-1].replace("。", "，")
            content = body + last_ch

        processed_parts.append({"tag": tag, "content": content})

    if not processed_parts:
        return ""

    merged_lines = []
    current_tag = processed_parts[0]["tag"]
    current_content = [processed_parts[0]["content"]]
    for part in processed_parts[1:]:
        if part["tag"] == current_tag and current_tag:
            current_content.append(part["content"])
        else:
            merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
            current_tag = part["tag"]
            current_content = [part["content"]]
    merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())

    return "".join(merged_lines).replace("‘", "'").replace("’", "'")


def validate_dialogue_text(dialogue_text: str, speaker_count: int) -> str:
    text = (dialogue_text or "").strip()
    if not text:
        raise ValueError("Please enter dialogue text.")

    tags = re.findall(r"\[S(\d+)\]", text)
    if not tags:
        raise ValueError("Dialogue must include speaker tags like [S1], [S2], ...")

    max_tag = max(int(t) for t in tags)
    if max_tag > speaker_count:
        raise ValueError(
            f"Dialogue contains [S{max_tag}], but speaker count is set to {speaker_count}."
        )
    return text


def merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
    if not segments:
        return text

    merged_parts = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if not matched:
            merged_parts.append(seg)
            continue
        tag, content = matched.groups()
        if tag == current_tag:
            merged_parts.append(content)
        else:
            current_tag = tag
            merged_parts.append(f"{tag}{content}")
    return "".join(merged_parts)


def normalize_prompt_text(prompt_text: str, speaker_id: int) -> str:
    text = (prompt_text or "").strip()
    if not text:
        raise ValueError(f"S{speaker_id} prompt text is empty.")

    expected_tag = f"[S{speaker_id}]"
    if not text.lstrip().startswith(expected_tag):
        text = f"{expected_tag} {text}"
    return text


def speakers(script: str, references: int, transcripts: int) -> int:
    """The speaker count of a job: the highest tag in the script, or more clips.

    The k-th reference clip (by file name) and the k-th transcript clone speaker
    ``[Sk]``; the creator requires both for a cloned speaker. Refuses before any load.
    """
    if references != transcripts:
        raise SystemExit(
            f"give one reference_text per reference_audio clip ({references} clips, "
            f"{transcripts} transcripts)"
        )
    tags = [int(tag) for tag in re.findall(r"\[S(\d+)\]", normalize_text(script))]
    count = max([*tags, references, MIN_SPEAKERS])
    if count > MAX_SPEAKERS:
        raise SystemExit(f"MOSS-TTSD takes at most {MAX_SPEAKERS} speakers, got {count}")
    try:
        validate_dialogue_text(normalize_text(script), count)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return count
