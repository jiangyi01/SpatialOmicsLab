"""Hear a visualisation need in what a user typed -- by rule, with no model call.

The portal recommends its explorer when a turn needs one, and this is the half that reads the words. It answers four
questions about one message:

``see``        did the user ask to LOOK at something -- "show the spatial distribution", "3D 展示一下", "make it
               interactive"? This opens the panel beside the chat without a click, so it needs a viewing verb AND a
               visual object in one sentence: "show that gene A is significant" asks for a proof, not a picture.
``three_d``    is the work three-dimensional -- "serial sections", "z-stack", "三维", "切片对齐"?
``flat_only``  did they ask for the flat view only -- "only 2D", "不要3D"? That clears ``three_d``.
``communication`` did they ask about cell-cell communication -- "ligand-receptor", "who signals to whom", "细胞通讯"?
               The card then offers the chat's communication result in the signalling view, or says to run an
               analysis first (Program 10). On its own it is a request for an analysis, not for the panel: it does
               not set ``see``, but it is a visual object for the ``see`` rule, so "show the cell-cell communication"
               and "展示信号流向" are asks to look and "run cell-cell communication with COMMOT" is not.

Rules rather than a model call, deliberately: no tokens, no latency, no rate limit to share with the turn, and no
change to the scored prompt. What the rules miss costs a click, not an answer -- the card still appears whenever the
turn's data can be drawn, which the explorer's own ``describe`` decides.

Left out on purpose, because this domain uses them for something else: bare "reconstruct"/"重建" (trajectory and
lineage reconstruction), "volume"/"体积", "depth" (sequencing depth), "堆叠" (a stacked bar chart). A 3D cue inside a
path or a file name (``/data/3d_stack.h5ad``) is not heard either, nor "3D printing"; nor a communication cue in one
(``commot_ligand-receptor_results.h5ad``), nor a bare "communication" ("the communication dataset").

Stdlib only: the portal imports this, and the portal must not import the analysis stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: The most of a message that is read. What the portal hands over is what the user TYPED -- a sentence or two -- so
#: a cue past this is in a pasted log or a document, not in the request.
MAX_CHARS = 4000
#: The most cues kept on an answer: enough to say why, short enough to carry in an event.
MAX_CUES = 8


@dataclass(frozen=True)
class VizIntent:
    see: bool = False
    three_d: bool = False
    flat_only: bool = False
    communication: bool = False
    cues: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, bool]:
        return {
            "see": self.see,
            "three_d": self.three_d,
            "flat_only": self.flat_only,
            "communication": self.communication,
        }


NO_INTENT = VizIntent()

_I = re.IGNORECASE

#: Paths and file names, removed before anything is matched: a 3D cue in ``3d_stack.h5ad`` is a name, not a request.
_PATHS = (re.compile(r"\S*[/\\]\S*"), re.compile(r"\S+\.[A-Za-z0-9]{1,6}\b"))
#: "3D printing" is about a printer.
_PRINTING = re.compile(r"3[- ]?d\s*(?:print\w*|打印)", _I)

_FLAT = [
    re.compile(p, _I)
    for p in (
        r"\b(?:only|just)\s+(?:in\s+)?2[- ]?d\b",
        r"\b2[- ]?d\s+only\b",
        r"\b(?:no|not\s+in|not|without)\s+3[- ]?d\b",
        r"只(?:要|看|用)\s*2[- ]?d",
        r"不(?:要|用)\s*(?:3[- ]?d|三维|立体)",
    )
]

_THREE_D = [
    re.compile(p, _I)
    for p in (
        r"(?<![A-Za-z0-9])3[- ]?d(?![A-Za-z0-9])",
        r"\bthree[- ]dimension(?:al|s)?\b",
        r"\bserial[- ]sections?\b",
        r"\bz[- ]?stacks?\b",
        r"\bz[- ]axis\b",
        r"\bstacks?\s+of\s+(?:\d+\s+)?(?:sections|slices)\b",
        r"\balign(?:ing|ed|ment)?\s+(?:the\s+|these\s+|all\s+)?(?:serial\s+)?(?:sections|slices)\b",
        r"\b(?:sections|slices)\s+align(?:ment|ed)?\b",
        r"\bvolumetric\b",
        r"三维",
        r"立体",
        r"连续切片",
        r"多切片",
        r"切片对齐",
        r"对齐切片",
        r"z\s*轴",
        r"切片堆叠",
    )
]

_SEE_VERB = [
    re.compile(
        r"\b(?:show(?:s|ing|n)?|display(?:s|ed|ing)?|visuali[sz](?:e|es|ed|ing|ation)|plot(?:s|ted|ting)?|"
        r"render(?:s|ed|ing)?|explor(?:e|es|ed|ing)|look(?:ing)?\s+at|let\s+me\s+see|view(?:s|ed|ing)?)\b",
        _I,
    ),
    re.compile(r"可视化|展示|显示|看看|看一下|查看|画|呈现|让我看"),
]
_SEE_OBJECT = [
    re.compile(
        r"\b(?:maps?|distributions?|clusters?|clustering|domains?|spots?|cells?|umap|t-?sne|embeddings?|tissues?|"
        r"sections?|slides?|slices?|results?|points?|patterns?|expression|layout|spatial(?:ly)?|3[- ]?d)\b",
        _I,
    ),
    re.compile(r"分布|聚类|结果|空间|结构域|区域|细胞|表达|切片|组织|图|umap|3d|三维|立体", _I),
]
#: An ask about cell-cell communication. Each is a phrase, not a word: bare "communication", "signal" or "interaction"
#: are everyday words ("the communication dataset", "signal-to-noise", "gene interactions").
_COMMUNICATION = [
    re.compile(p, _I)
    for p in (
        r"\b(?:cell[- ]?(?:to[- ])?cell|intercellular)\s+(?:communication|interactions?|signal(?:l?ing)?)\b",
        r"\bligand[- ]receptor\b",
        r"\bL-?R\s+pairs?\b",
        r"\bwho\s+(?:signals|talks)\s+to\s+whom\b",
        r"\bsenders?\s+and\s+receivers?\b",
        r"\bcommunication\s+(?:map|network|direction|flow)\b",
        r"\bsignal(?:l?ing)?\s+(?:flow|direction)\b",
        r"\bcrosstalk\b",
        r"细胞(?:间)?(?:通讯|通信|互作|相互作用)",
        r"配体[- ]?受体",
        r"信号(?:传导|流|方向|流向)",
        r"谁在向谁",
    )
]
#: An interaction word is an ask to look on its own.
_INTERACT = [
    re.compile(r"\binteractive(?:ly)?\b|\bzoom(?:s|ed|ing)?\b|\bpan\s+around\b", _I),
    re.compile(r"可交互|交互|缩放|拖动"),
]
#: Rotation asks to look only at something three-dimensional: rotating an H&E image is not exploring it.
_ROTATE = re.compile(r"\brotat(?:e|es|ed|ing|ion)\b|旋转", _I)
_SENTENCE = re.compile(r"[.?!。？！;；\n]+")


def _hits(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    return [m.group(0).strip().lower() for p in patterns for m in p.finditer(text)]


def read_viz_intent(text: Any) -> VizIntent:
    """The visualisation need ``text`` states, or :data:`NO_INTENT`. Never raises; anything but ``str`` is nothing."""
    if not isinstance(text, str) or not text.strip():
        return NO_INTENT
    try:
        body = text[:MAX_CHARS]
        for path in _PATHS:
            body = path.sub(" ", body)
        body = _PRINTING.sub(" ", body)

        cues: list[str] = []
        flat = _hits(_FLAT, body)
        three = [] if flat else _hits(_THREE_D, body)

        talk = _hits(_COMMUNICATION, body)

        see: list[str] = []
        for sentence in _SENTENCE.split(body):
            verbs = _hits(_SEE_VERB, sentence)
            # A communication cue is something to look at: "show the signalling flow" names no other object.
            if verbs and (_hits(_SEE_OBJECT, sentence) or _hits(_COMMUNICATION, sentence)):
                see.extend(verbs)
        see.extend(_hits(_INTERACT, body))
        if three:
            see.extend(_hits([_ROTATE], body))

        for cue in [*see, *three, *flat, *talk]:
            if cue and cue not in cues:
                cues.append(cue)
        if not cues:
            return NO_INTENT
        return VizIntent(
            see=bool(see),
            three_d=bool(three),
            flat_only=bool(flat),
            communication=bool(talk),
            cues=tuple(cues[:MAX_CUES]),
        )
    except Exception:
        return NO_INTENT
