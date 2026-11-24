"""Render docs/demo.gif from a real run of `ragdial-demo`.

The GIF is not a mock-up: this script runs the command with colour forced on,
captures its actual ANSI output, and animates it as a terminal session (typed
command, output streamed step by step, final hold). Colours, bold text and
symbols are drawn exactly as the CLI emits them.

    uv run --with pillow python scripts/make_demo_gif.py                 # local checkout
    uv run --with pillow python scripts/make_demo_gif.py --cmd "uvx --from ragdial-mcp ragdial-demo"
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

SURFACE = (26, 26, 25)        # #1a1a19, the dark surface the CLI palette was validated on
TITLE_BAR = (35, 35, 33)
INK = (240, 239, 236)
MUTED = (138, 137, 132)
PROMPT = (12, 163, 12)
DOTS = [(237, 106, 94), (245, 191, 79), (98, 197, 84)]

ANSI16 = {
    30: (0, 0, 0), 31: (208, 59, 59), 32: (12, 163, 12), 33: (201, 133, 0), 34: (57, 135, 229),
    35: (213, 81, 129), 36: (25, 158, 112), 37: (195, 194, 183), 90: (138, 137, 132), 91: (230, 103, 103),
    92: (12, 163, 12), 93: (250, 178, 25), 94: (57, 135, 229), 95: (213, 81, 129), 96: (25, 158, 112),
    97: (240, 239, 236),
}

REGULAR = ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
           "/System/Library/Fonts/Menlo.ttc"]
BOLD = ["C:/Windows/Fonts/consolab.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc"]
SYMBOL = ["C:/Windows/Fonts/seguisym.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "/System/Library/Fonts/Apple Symbols.ttf"]
EIGHTHS = "▏▎▍▌▋▊▉"


def first_font(paths: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    sys.exit(f"No font found among {paths}")


@dataclass
class Seg:
    text: str
    fg: tuple[int, int, int]
    bg: tuple[int, int, int] | None
    bold: bool


SGR = re.compile(r"\x1b\[([0-9;]*)m")


def parse_ansi(line: str) -> list[Seg]:
    segs: list[Seg] = []
    fg, bg, bold = INK, None, False
    pos = 0
    for m in SGR.finditer(line):
        if m.start() > pos:
            segs.append(Seg(line[pos:m.start()], fg, bg, bold))
        codes = [int(c) for c in m.group(1).split(";") if c] or [0]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == 0:
                fg, bg, bold = INK, None, False
            elif c == 1:
                bold = True
            elif c == 22:
                bold = False
            elif c == 39:
                fg = INK
            elif c == 49:
                bg = None
            elif c in (38, 48) and i + 4 < len(codes) and codes[i + 1] == 2:
                rgb = (codes[i + 2], codes[i + 3], codes[i + 4])
                if c == 38:
                    fg = rgb
                else:
                    bg = rgb
                i += 4
            elif c in (38, 48) and i + 2 < len(codes) and codes[i + 1] == 5:
                i += 2  # 256-colour codes are not emitted with COLORTERM=truecolor
            elif c in ANSI16:
                fg = ANSI16[c]
            i += 1
        pos = m.end()
    if pos < len(line):
        segs.append(Seg(line[pos:], fg, bg, bold))
    return segs


def run_demo(cmd: str, cols: int) -> list[list[Seg]]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "FORCE_COLOR": "1", "COLORTERM": "truecolor",
           "TERM": "xterm-256color", "COLUMNS": str(cols)}
    env.pop("NO_COLOR", None)
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(shlex.split(cmd) + ["--home", "runs"], cwd=tmp, env=env, check=True,
                             capture_output=True, text=True, encoding="utf-8").stdout
    lines = out.rstrip("\n").split("\n")
    lines = [re.sub(r"Runs stored in .*", "Runs stored in ./runs", ln) for ln in lines]  # never leak local paths
    return [parse_ansi(ln.rstrip()) for ln in lines]


class Terminal:
    def __init__(self, cols: int, rows: int, size: int, title: str) -> None:
        self.reg, self.bold, self.sym = first_font(REGULAR, size), first_font(BOLD, size), first_font(SYMBOL, size)
        self.cw = self.reg.getlength("M")
        asc, desc = self.reg.getmetrics()
        self.ascent = asc
        self.lh = int((asc + desc) * 1.18)
        self.pad, self.bar = 20, 36
        self.cols, self.rows, self.title = cols, rows, title
        self.w = int(self.cw * cols + self.pad * 2)
        self.h = self.bar + self.lh * rows + self.pad * 2
        # each font has its own "missing glyph" box; compare against the right one
        self._notdef = {id(f): self._mask(f, "\uffff") for f in (self.reg, self.bold)}

    @staticmethod
    def _mask(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
        m = font.getmask(ch)
        return bytes(m) if m.size[0] else b""

    @lru_cache(maxsize=None)
    def font_for(self, ch: str, bold: bool) -> ImageFont.FreeTypeFont:
        f = self.bold if bold else self.reg
        if ord(ch) < 0x2000 or self._mask(f, ch) != self._notdef[id(f)]:
            return f
        return self.sym

    def frame(self, lines: list[list[Seg]], cursor: bool) -> Image.Image:
        img = Image.new("RGB", (self.w, self.h), SURFACE)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, self.w, self.bar], fill=TITLE_BAR)
        for i, c in enumerate(DOTS):
            x = 18 + i * 20
            d.ellipse([x, 12, x + 12, 24], fill=c)
        tw = self.reg.getlength(self.title)
        d.text(((self.w - tw) / 2, 10), self.title, font=self.reg, fill=MUTED)
        visible = lines[-self.rows:]
        y = self.bar + self.pad
        col = 0
        for segs in visible:
            col = 0
            for seg in segs:
                for ch in seg.text:
                    x = self.pad + col * self.cw
                    if seg.bg:
                        d.rectangle([x, y - 1, x + self.cw + 1, y + self.lh - 2], fill=seg.bg)
                    if ch == "█":  # draw blocks as exact cells so bars are seamless
                        d.rectangle([x, y + 3, x + self.cw, y + self.lh - 5], fill=seg.fg)
                    elif ch in EIGHTHS:
                        frac = EIGHTHS.index(ch) + 1
                        d.rectangle([x, y + 3, x + self.cw * frac / 8, y + self.lh - 5], fill=seg.fg)
                    elif ch != " ":
                        # shared baseline, so fallback-font symbols line up with the text
                        d.text((x, y + self.ascent), ch, font=self.font_for(ch, seg.bold), fill=seg.fg,
                               anchor="ls")
                    col += 1
            y += self.lh
        if cursor:
            cx = self.pad + col * self.cw
            cy = self.bar + self.pad + self.lh * (len(visible) - 1)
            d.rectangle([cx + 1, cy + 2, cx + self.cw - 1, cy + self.lh - 4], fill=INK)
        return img


STEP_HEADER = re.compile(r"^ \d ")


def build(cmd_shown: str, output: list[list[Seg]], term: Terminal) -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    shown: list[list[Seg]] = []

    def push(ms: int, cursor: bool = False) -> None:
        frames.append(term.frame(shown, cursor))
        durations.append(ms)

    prompt = Seg("$ ", PROMPT, None, True)
    shown.append([prompt])
    push(700, cursor=True)
    for i in range(2, len(cmd_shown) + 2, 2):
        shown[-1] = [prompt, Seg(cmd_shown[:i], INK, None, False)]
        push(45, cursor=True)
    push(600, cursor=True)

    for segs in output:
        shown.append(segs)
        plain = "".join(s.text for s in segs)
        if STEP_HEADER.match(plain):
            push(1100)          # pause on each step header
        elif not plain.strip():
            push(200)
        else:
            push(90)
    push(7000)
    return frames, durations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="uv run --project " + shlex.quote(ROOT.as_posix()) + " ragdial-demo",
                    help="Command that produces the output (default: this checkout)")
    ap.add_argument("--show", default="uvx --from ragdial-mcp ragdial-demo", help="Command shown being typed")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--rows", type=int, default=None, help="Terminal height (default: fit the whole run)")
    ap.add_argument("--font-size", type=int, default=15)
    args = ap.parse_args()

    output = run_demo(args.cmd, args.cols)
    rows = args.rows or len(output) + 1  # +1 for the typed command line
    term = Terminal(args.cols, rows, args.font_size, "ragdial-demo")
    frames, durations = build(args.show, output, term)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # one shared palette (from the final, fullest frame) so colours never flicker between frames
    ref = frames[-1].quantize(colors=128)
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"{out}: {len(frames)} frames, {out.stat().st_size / 1024:.0f} KiB, {term.w}x{term.h}px, "
          f"{sum(durations) / 1000:.1f}s")


if __name__ == "__main__":
    main()
