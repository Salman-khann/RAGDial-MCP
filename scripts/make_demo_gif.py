"""Render docs/demo.gif from a real run of `ragdial-demo`.

The GIF is not a mock-up: this script runs the command, captures its actual
stdout, and animates it as a terminal session (typed command, output streamed
section by section, final hold).

    uv run --with pillow python scripts/make_demo_gif.py
    uv run --with pillow python scripts/make_demo_gif.py --cmd "uv run ragdial-demo"
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

BG = (13, 17, 23)
BAR = (22, 27, 34)
FG = (201, 209, 217)
DIM = (110, 118, 129)
BLUE = (88, 166, 255)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
YELLOW = (210, 153, 34)
DOTS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Menlo.ttc",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    sys.exit("No monospace font found; add one to FONT_CANDIDATES")


def run_demo(cmd: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}
        out = subprocess.run(shlex.split(cmd) + ["--home", "runs"], cwd=tmp, env=env, check=True,
                             capture_output=True, text=True, encoding="utf-8").stdout
    lines = out.rstrip("\n").splitlines()
    # never leak local paths into a public asset
    return [re.sub(r"Runs stored in .*", "Runs stored in ./runs", ln) for ln in lines]


def wrap(lines: list[str], cols: int) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if len(ln) <= cols:
            out.append(ln)
        else:
            indent = len(ln) - len(ln.lstrip())
            out.extend(textwrap.wrap(ln, cols, subsequent_indent=" " * (indent + 4),
                                     drop_whitespace=False, break_on_hyphens=False) or [""])
    return out


def color_for(line: str) -> tuple[int, int, int]:
    s = line.strip()
    if re.match(r"^\[\d\]", s):
        return BLUE
    if set(s) <= set("- ") and s:
        return DIM
    if s.startswith(("fixed:", "regressed: []")) or "improves" in s:
        return GREEN
    if s.startswith("Done in"):
        return DIM
    if s.startswith("config ") or s.startswith("variant "):
        return YELLOW
    return FG


class Terminal:
    def __init__(self, cols: int, rows: int, font_size: int, title: str) -> None:
        self.font = load_font(font_size)
        box = self.font.getbbox("M")
        self.cw = self.font.getlength("M")
        self.lh = int((box[3] - box[1]) * 1.55)
        self.pad = 18
        self.bar = 34
        self.cols, self.rows = cols, rows
        self.w = int(self.cw * cols + self.pad * 2)
        self.h = self.bar + self.lh * rows + self.pad * 2
        self.title = title

    def frame(self, lines: list[tuple[str, tuple[int, int, int]]], cursor: bool) -> Image.Image:
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, self.w, self.bar], fill=BAR)
        for i, c in enumerate(DOTS):
            x = 18 + i * 20
            d.ellipse([x, 11, x + 12, 23], fill=c)
        tw = self.font.getlength(self.title)
        d.text(((self.w - tw) / 2, 9), self.title, font=self.font, fill=DIM)
        visible = lines[-self.rows:]
        y = self.bar + self.pad
        for text, color in visible:
            x = self.pad
            # highlight the cut marker inline
            for part in re.split(r"(<CUT>)", text):
                if not part:
                    continue
                d.text((x, y), part, font=self.font, fill=RED if part == "<CUT>" else color)
                x += self.font.getlength(part)
            y += self.lh
        if cursor:
            last = visible[-1][0] if visible else ""
            cx = self.pad + self.font.getlength(last)
            cy = self.bar + self.pad + self.lh * (len(visible) - 1)
            d.rectangle([cx + 2, cy + 2, cx + self.cw, cy + self.lh - 4], fill=FG)
        return img


def build(cmd_shown: str, output: list[str], term: Terminal) -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    shown: list[tuple[str, tuple[int, int, int]]] = []

    def push(ms: int, cursor: bool = False) -> None:
        frames.append(term.frame(shown, cursor))
        durations.append(ms)

    prompt = "$ "
    shown.append((prompt, GREEN))
    push(700, cursor=True)
    for i in range(1, len(cmd_shown) + 1, 2):  # type two chars per frame
        shown[-1] = (prompt + cmd_shown[:i + 1], FG)
        push(45, cursor=True)
    push(500, cursor=True)

    for line in wrap(output, term.cols):
        shown.append((line, color_for(line)))
        s = line.strip()
        if re.match(r"^\[\d\]", s):
            push(900)  # pause on each section header
        elif not s:
            push(250)
        else:
            push(110)
    push(6000)
    return frames, durations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="uvx --from ragdial-mcp ragdial-demo",
                    help="Command to run (and to show being typed)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    ap.add_argument("--cols", type=int, default=104)
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--font-size", type=int, default=15)
    args = ap.parse_args()

    output = run_demo(args.cmd)
    term = Terminal(args.cols, args.rows, args.font_size, "ragdial-demo")
    frames, durations = build(args.cmd, output, term)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # one shared palette (from the final, fullest frame) so colours never flicker between frames
    ref = frames[-1].quantize(colors=128)
    palette = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    palette[0].save(out, save_all=True, append_images=palette[1:], duration=durations, loop=0,
                    optimize=True, disposal=1)
    print(f"{out}: {len(frames)} frames, {out.stat().st_size / 1024:.0f} KiB, {term.w}x{term.h}px")


if __name__ == "__main__":
    main()
