#!/usr/bin/env python3
"""
chess_pgn_scanner.py — Convert a chess scoresheet photo into a legal PGN.

Uses Claude vision to read moves, python-chess to validate legality,
and a tkinter GUI to interactively correct any illegal moves.

Usage:
    python chess_pgn_scanner.py <image_path> [output.pgn]
"""

import anthropic
import chess
import chess.pgn
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import base64
import io
import json
import re
import sys
from datetime import date
from pathlib import Path

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_SUPPORTED = True
except ImportError:
    _HEIC_SUPPORTED = False


# ── Unicode pieces ────────────────────────────────────────────────────────────

PIECE_UNICODE = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}


# ── Image encoding ────────────────────────────────────────────────────────────

MAX_BYTES = 5 * 1024 * 1024  # Claude's 5 MB limit

def _encode_jpeg_under_limit(img: Image.Image) -> str:
    """Encode a PIL image as JPEG base64, scaling down until it fits under 5 MB."""
    for quality in (92, 80, 65):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_BYTES:
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # Still too large — shrink resolution by half and retry
    small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=80)
    print(f"  Image resized to {small.width}×{small.height} to fit 5 MB API limit.")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a given image file.

    HEIC files are converted to JPEG in memory because the Claude API
    does not accept the HEIC format directly.
    """
    suffix = Path(path).suffix.lower()

    if suffix in (".heic", ".heif"):
        if not _HEIC_SUPPORTED:
            raise RuntimeError(
                "pillow-heif is required for HEIC files.\n"
                "Install it with:  pip3 install pillow-heif"
            )
        img = Image.open(path).convert("RGB")
        data = _encode_jpeg_under_limit(img)
        return data, "image/jpeg"

    media_type = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


# ── Claude OCR ────────────────────────────────────────────────────────────────

def extract_moves_from_image(image_path: str) -> list[str]:
    """Ask Claude to read all chess moves from a scoresheet image."""
    client = anthropic.Anthropic()
    image_data, media_type = encode_image(image_path)

    prompt = (
        "Please examine this chess scoresheet carefully.\n\n"
        "Extract every chess move you can see, in order (White move, Black move, …).\n"
        "Return ONLY a JSON array of moves in standard algebraic notation, e.g.:\n"
        '["e4","e5","Nf3","Nc6","Bb5"]\n\n'
        "Include both colours in sequence. If a move is unclear, give your best guess.\n"
        "Do NOT include move numbers. Return ONLY the JSON array — no other text."
    )

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = message.content[0].text.strip()
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            moves = json.loads(match.group())
            return [str(m).strip() for m in moves if m]
        except json.JSONDecodeError:
            pass

    print(f"Warning: could not parse Claude's response as JSON:\n{text}")
    return []


# ── OCR error correction ──────────────────────────────────────────────────────

_OCR_FIXES = [
    (r"^0-0-0$", "O-O-O"),
    (r"^0-0$",   "O-O"),
    (r"([a-h])l",  r"\g<1>1"),   # lowercase-L → digit 1 after a file letter
    (r"([a-h])I",  r"\g<1>1"),   # uppercase-I → digit 1 after a file letter
    (r"\bO\b",    "0"),           # stray capital O sometimes means 0 in coordinates
]

def fix_ocr_errors(move: str) -> str:
    result = move
    for pattern, repl in _OCR_FIXES:
        result = re.sub(pattern, repl, result)
    return result


# ── Move suggestion ───────────────────────────────────────────────────────────

def suggest_similar_moves(board: chess.Board, ocr_move: str, n: int = 6) -> list[str]:
    """Return up to n legal moves most similar to the OCR'd text."""
    legal_sans = [board.san(m) for m in board.legal_moves]

    def score(san: str) -> int:
        s = 0
        ol, sl = ocr_move.lower(), san.lower()
        s += sum(1 for c in ol if c in sl) * 2
        s -= abs(len(san) - len(ocr_move))
        # Same destination square (last two chars)?
        if len(ocr_move) >= 2 and ocr_move[-2:].lower() in sl:
            s += 5
        # Same piece letter?
        if ocr_move and ocr_move[0].isupper() and san and san[0] == ocr_move[0]:
            s += 4
        return s

    ranked = sorted(legal_sans, key=score, reverse=True)
    return ranked[:n]


# ── Board canvas widget ───────────────────────────────────────────────────────

class BoardCanvas(tk.Canvas):
    SQ   = 56
    LIGHT = "#F0D9B5"
    DARK  = "#B58863"
    HILITE = "#AAD751"
    HILITE_DARK = "#86A730"

    def __init__(self, parent, board: chess.Board, last_move=None, **kw):
        size = self.SQ * 8
        super().__init__(parent, width=size, height=size, **kw)
        self.board = board
        self.redraw(last_move)

    def redraw(self, last_move=None):
        self.delete("all")
        ss = self.SQ
        hi = set()
        if last_move:
            hi = {last_move.from_square, last_move.to_square}

        for rank in range(7, -1, -1):
            for file in range(8):
                sq = chess.square(file, rank)
                x, y = file * ss, (7 - rank) * ss
                light = (file + rank) % 2 != 0
                if sq in hi:
                    color = self.HILITE if light else self.HILITE_DARK
                else:
                    color = self.LIGHT if light else self.DARK
                self.create_rectangle(x, y, x + ss, y + ss, fill=color, outline="")

                piece = self.board.piece_at(sq)
                if piece:
                    sym = PIECE_UNICODE.get((piece.piece_type, piece.color), "?")
                    fg  = "white" if piece.color == chess.WHITE else "#1a1a1a"
                    ol  = "#1a1a1a" if piece.color == chess.WHITE else "white"
                    cx, cy = x + ss // 2, y + ss // 2
                    fs = int(ss * 0.60)
                    for dx, dy in ((-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0)):
                        self.create_text(cx+dx, cy+dy, text=sym,
                                         font=("Arial", fs), fill=ol)
                    self.create_text(cx, cy, text=sym, font=("Arial", fs), fill=fg)

        # coordinate labels
        for i in range(8):
            self.create_text(i*ss + ss//2, 8*ss - 4,
                             text="abcdefgh"[i], font=("Arial", 8), fill="#888")
            self.create_text(4, (7-i)*ss + ss//2,
                             text=str(i+1), font=("Arial", 8), fill="#888")


# ── Correction dialog ─────────────────────────────────────────────────────────

class CorrectionDialog:
    """
    Pop-up dialog that shows the scoresheet image alongside the current board
    and lets the user pick or type the correct move.
    """

    BG = "#1e1e2e"
    FG = "#cdd6f4"

    def __init__(self, image_path: str, board: chess.Board,
                 move_label: str, ocr_move: str, suggestions: list[str],
                 current_raw_idx: int = 0):
        # result is (raw_move_index, san) or None to abort
        self.result: tuple[int, str] | None = None
        self._image_path = image_path
        self._board = board.copy()
        self._move_label = move_label
        self._ocr_move = ocr_move
        self._suggestions = suggestions
        self._current_raw_idx = current_raw_idx
        self._zoom = 1.0
        self._orig_img: Image.Image | None = None

        # Build position history: list of (board_snapshot, last_move, san)
        # Index 0 = starting position, last index = current position
        self._history: list[tuple[chess.Board, chess.Move | None, str]] = []
        b = chess.Board()
        self._history.append((b.copy(), None, "Start"))
        for move in board.move_stack:
            san = b.san(move)
            b.push(move)
            self._history.append((b.copy(), move, san))
        self._view_idx = len(self._history) - 1   # start at current position

        self._run()

    # ── UI construction ───────────────────────────────────────────────────────

    def _run(self):
        root = tk.Tk()
        root.title("Move Correction")
        root.configure(bg=self.BG)
        root.resizable(True, True)

        self._root = root
        self._build(root)

        # Maximize to full screen size
        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{sw}x{sh}+0+0")

        # After the window is drawn, fit the image to whatever space it actually has
        root.after(100, self._fit_to_canvas)

        self._entry.focus_set()
        root.mainloop()

    def _build(self, root):
        pad = dict(padx=10, pady=6)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=self.BG)
        hdr.pack(fill=tk.X, **pad)

        tk.Label(hdr,
                 text=f"Illegal move detected  —  {self._move_label}: \"{self._ocr_move}\"",
                 font=("Arial", 14, "bold"), fg="#f38ba8", bg=self.BG).pack(anchor="w")
        tk.Label(hdr,
                 text="Select a suggestion or type the correct move below.",
                 font=("Arial", 11), fg="#a6adc8", bg=self.BG).pack(anchor="w")

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=10)

        # ── Side-by-side panels ───────────────────────────────────────────────
        panels = tk.Frame(root, bg=self.BG)
        panels.pack(fill=tk.BOTH, expand=True, **pad)

        # — Scoresheet image —
        img_lf = tk.LabelFrame(panels, text=" Scoresheet ",
                               font=("Arial", 10, "bold"),
                               fg=self.FG, bg=self.BG, bd=1, relief="groove")
        img_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        try:
            self._orig_img = Image.open(self._image_path)
            # Normalise orientation from EXIF
            self._orig_img = self._orig_img.convert("RGB")

            # Scrollable canvas for the image
            CANVAS_W, CANVAS_H = 420, 500
            img_canvas = tk.Canvas(img_lf, width=CANVAS_W, height=CANVAS_H,
                                   bg=self.BG, highlightthickness=0)
            vsb = tk.Scrollbar(img_lf, orient="vertical",   command=img_canvas.yview)
            hsb = tk.Scrollbar(img_lf, orient="horizontal", command=img_canvas.xview)
            img_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            vsb.pack(side=tk.RIGHT,  fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)
            img_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            self._img_canvas = img_canvas

            # Zoom controls
            zoom_bar = tk.Frame(img_lf, bg=self.BG)
            zoom_bar.pack(fill=tk.X, padx=4, pady=(0, 4))
            tk.Button(zoom_bar, text="−", font=("Arial", 13, "bold"),
                      bg="#45475a", fg=self.FG, relief="flat", width=3,
                      cursor="hand2", command=self._zoom_out).pack(side=tk.LEFT)
            tk.Button(zoom_bar, text="+", font=("Arial", 13, "bold"),
                      bg="#45475a", fg=self.FG, relief="flat", width=3,
                      cursor="hand2", command=self._zoom_in).pack(side=tk.LEFT, padx=4)
            self._zoom_label = tk.Label(zoom_bar, text="100%",
                                        font=("Arial", 10), fg="#a6adc8", bg=self.BG)
            self._zoom_label.pack(side=tk.LEFT)
            tk.Label(zoom_bar, text="  scroll wheel also works",
                     font=("Arial", 9, "italic"), fg="#585b70", bg=self.BG).pack(side=tk.LEFT)

            # Mouse-wheel zoom
            img_canvas.bind("<MouseWheel>",      self._on_mousewheel)   # macOS / Windows
            img_canvas.bind("<Button-4>",        self._on_mousewheel)   # Linux scroll up
            img_canvas.bind("<Button-5>",        self._on_mousewheel)   # Linux scroll down

            self._render_image()

        except Exception as exc:
            tk.Label(img_lf, text=f"(Could not load image)\n{exc}",
                     fg="#f38ba8", bg=self.BG, wraplength=200).pack(padx=8, pady=8)

        # — Board + navigation —
        right = tk.Frame(panels, bg=self.BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH)

        board_lf = tk.LabelFrame(right, text=" Position ",
                                 font=("Arial", 10, "bold"),
                                 fg=self.FG, bg=self.BG, bd=1, relief="groove")
        board_lf.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))

        self._board_canvas = BoardCanvas(board_lf, self._history[self._view_idx][0],
                                         last_move=self._history[self._view_idx][1],
                                         bg=self.BG)
        self._board_canvas.pack(padx=6, pady=6)

        # Nav bar: ← position indicator →
        nav = tk.Frame(board_lf, bg=self.BG)
        nav.pack(fill=tk.X, padx=6, pady=(0, 4))

        tk.Button(nav, text="◀◀", font=("Arial", 11), width=3,
                  bg="#45475a", fg=self.FG, relief="flat", cursor="hand2",
                  command=self._go_start).pack(side=tk.LEFT)
        tk.Button(nav, text="◀", font=("Arial", 11), width=3,
                  bg="#45475a", fg=self.FG, relief="flat", cursor="hand2",
                  command=self._go_prev).pack(side=tk.LEFT, padx=2)

        self._pos_label = tk.Label(nav, text="", font=("Arial", 9),
                                   fg="#a6adc8", bg=self.BG, width=14)
        self._pos_label.pack(side=tk.LEFT, padx=4)

        tk.Button(nav, text="▶", font=("Arial", 11), width=3,
                  bg="#45475a", fg=self.FG, relief="flat", cursor="hand2",
                  command=self._go_next).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="▶▶", font=("Arial", 11), width=3,
                  bg="#45475a", fg=self.FG, relief="flat", cursor="hand2",
                  command=self._go_end).pack(side=tk.LEFT)

        # Keyboard shortcuts
        self._root.bind("<Left>",  lambda _: self._go_prev())
        self._root.bind("<Right>", lambda _: self._go_next())
        self._root.bind("<Home>",  lambda _: self._go_start())
        self._root.bind("<End>",   lambda _: self._go_end())

        # — Move list —
        ml_lf = tk.LabelFrame(right, text=" Moves ",
                               font=("Arial", 10, "bold"),
                               fg=self.FG, bg=self.BG, bd=1, relief="groove")
        ml_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ml_scroll = tk.Scrollbar(ml_lf, orient="vertical")
        self._move_list = tk.Listbox(ml_lf, font=("Courier", 11),
                                     bg="#181825", fg=self.FG,
                                     selectbackground="#585b70",
                                     selectforeground="white",
                                     activestyle="none",
                                     width=16, height=14,
                                     yscrollcommand=ml_scroll.set,
                                     exportselection=False)
        ml_scroll.config(command=self._move_list.yview)
        ml_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._move_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._move_list.bind("<<ListboxSelect>>", self._on_list_select)

        # Per-move edit box (shown when a move row is selected)
        ml_edit = tk.Frame(ml_lf, bg=self.BG)
        ml_edit.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._list_edit_var = tk.StringVar()
        self._list_entry = tk.Entry(ml_edit, textvariable=self._list_edit_var,
                                    font=("Courier", 11), width=8,
                                    bg="#313244", fg=self.FG,
                                    insertbackground=self.FG,
                                    relief="flat", bd=3)
        self._list_entry.pack(side=tk.LEFT)
        self._list_entry.bind("<Return>", self._apply_to_selected)

        self._apply_btn = tk.Button(ml_edit, text="Apply",
                                    font=("Arial", 10), bg="#cba6f7", fg="#1e1e2e",
                                    relief="flat", padx=6, pady=2, cursor="hand2",
                                    command=self._apply_to_selected)
        self._apply_btn.pack(side=tk.LEFT, padx=4)

        self._list_hint = tk.Label(ml_edit, text="← click a move to edit",
                                   font=("Arial", 9, "italic"), fg="#585b70", bg=self.BG)
        self._list_hint.pack(side=tk.LEFT)

        self._populate_move_list()
        self._refresh_board_view()

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=10)

        # ── Suggestions ───────────────────────────────────────────────────────
        sug_frame = tk.Frame(root, bg=self.BG)
        sug_frame.pack(fill=tk.X, **pad)

        tk.Label(sug_frame, text="Suggested moves:",
                 font=("Arial", 11, "bold"), fg=self.FG, bg=self.BG).pack(anchor="w")

        btn_row = tk.Frame(sug_frame, bg=self.BG)
        btn_row.pack(anchor="w", pady=4)

        for san in self._suggestions:
            tk.Button(btn_row, text=san,
                      font=("Arial", 12, "bold"),
                      bg="#45475a", fg=self.FG, activebackground="#585b70",
                      relief="flat", padx=10, pady=6, cursor="hand2",
                      command=lambda m=san: self._pick(m)).pack(side=tk.LEFT, padx=3)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=10)

        # ── Manual entry ──────────────────────────────────────────────────────
        entry_frame = tk.Frame(root, bg=self.BG)
        entry_frame.pack(fill=tk.X, **pad)

        tk.Label(entry_frame, text="Enter move manually:",
                 font=("Arial", 11), fg=self.FG, bg=self.BG).pack(side=tk.LEFT)

        self._var = tk.StringVar()
        self._entry = tk.Entry(entry_frame, textvariable=self._var,
                               font=("Arial", 13), width=10,
                               bg="#313244", fg=self.FG, insertbackground=self.FG,
                               relief="flat", bd=4)
        self._entry.pack(side=tk.LEFT, padx=8)
        self._entry.bind("<Return>", self._manual_submit)

        tk.Button(entry_frame, text="Submit",
                  font=("Arial", 11), bg="#a6e3a1", fg="#1e1e2e",
                  relief="flat", padx=10, pady=4, cursor="hand2",
                  command=self._manual_submit).pack(side=tk.LEFT)

        tk.Button(entry_frame, text="Abort",
                  font=("Arial", 11), bg="#f38ba8", fg="#1e1e2e",
                  relief="flat", padx=10, pady=4, cursor="hand2",
                  command=self._abort).pack(side=tk.RIGHT)

        self._root.protocol("WM_DELETE_WINDOW", self._abort)

    # ── Navigation helpers ────────────────────────────────────────────────────

    def _populate_move_list(self):
        """One row per move: row index == history index."""
        self._move_list.delete(0, tk.END)
        self._move_list.insert(tk.END, "  Start")
        for i, (_, _, san) in enumerate(self._history[1:], start=1):
            num = (i + 1) // 2
            dot = "." if i % 2 == 1 else "…"
            # Mark the current illegal move distinctly
            marker = " ◀ ?" if i - 1 == self._current_raw_idx else ""
            self._move_list.insert(tk.END, f"  {num}{dot} {san}{marker}")

    def _refresh_board_view(self):
        snap, last_move, san = self._history[self._view_idx]
        self._board_canvas.board = snap
        self._board_canvas.redraw(last_move)

        if self._view_idx == 0:
            label = "Start"
        else:
            num = (self._view_idx + 1) // 2
            dot = "." if self._view_idx % 2 == 1 else "…"
            label = f"{num}{dot} {san}"
        self._pos_label.configure(text=label)

        self._move_list.selection_clear(0, tk.END)
        self._move_list.selection_set(self._view_idx)
        self._move_list.see(self._view_idx)

    def _go_prev(self):
        if self._view_idx > 0:
            self._view_idx -= 1
            self._refresh_board_view()

    def _go_next(self):
        if self._view_idx < len(self._history) - 1:
            self._view_idx += 1
            self._refresh_board_view()

    def _go_start(self):
        self._view_idx = 0
        self._refresh_board_view()

    def _go_end(self):
        self._view_idx = len(self._history) - 1
        self._refresh_board_view()

    def _on_list_select(self, _event):
        sel = self._move_list.curselection()
        if not sel:
            return
        self._view_idx = sel[0]
        self._refresh_board_view()
        # Populate the edit box with this move's SAN (not for "Start" row)
        if self._view_idx > 0:
            san = self._history[self._view_idx][2]
            self._list_edit_var.set(san)
            self._list_entry.focus_set()
            self._list_entry.selection_range(0, tk.END)
            self._list_hint.configure(text=f"editing move {self._view_idx}")
        else:
            self._list_edit_var.set("")
            self._list_hint.configure(text="← click a move to edit")

    # ── Zoom helpers ──────────────────────────────────────────────────────────

    def _fit_to_canvas(self):
        """Set zoom so the image fills the canvas without overflowing."""
        if self._orig_img is None:
            return
        cw = self._img_canvas.winfo_width()
        ch = self._img_canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        self._zoom = min(1.0, cw / self._orig_img.width, ch / self._orig_img.height)
        self._render_image()

    def _render_image(self):
        if self._orig_img is None:
            return
        w = int(self._orig_img.width  * self._zoom)
        h = int(self._orig_img.height * self._zoom)
        resized = self._orig_img.resize((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self._img_canvas.delete("all")
        self._img_canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._img_canvas.configure(scrollregion=(0, 0, w, h))
        self._zoom_label.configure(text=f"{int(self._zoom * 100)}%")

    def _zoom_in(self):
        self._zoom = min(self._zoom * 1.25, 5.0)
        self._render_image()

    def _zoom_out(self):
        self._zoom = max(self._zoom / 1.25, 0.2)
        self._render_image()

    def _on_mousewheel(self, event):
        # macOS: event.delta is ±120 multiples; Linux uses Button-4/5
        if event.num == 4 or event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _pick(self, san: str):
        try:
            self._board.parse_san(san)
            self.result = (self._current_raw_idx, san)
            self._root.destroy()
        except Exception:
            pass

    def _manual_submit(self, _event=None):
        san = self._var.get().strip()
        try:
            self._board.parse_san(san)
            self.result = (self._current_raw_idx, san)
            self._root.destroy()
        except Exception:
            self._entry.configure(bg="#45232e")
            self._root.after(400, lambda: self._entry.configure(bg="#313244"))

    def _apply_to_selected(self, _event=None):
        """Apply the list-edit box value to whichever move is selected in the list."""
        if self._view_idx == 0:
            return   # can't edit "Start"
        san = self._list_edit_var.get().strip()
        # Validate against the board position *before* this move
        board_before = self._history[self._view_idx - 1][0]
        try:
            board_before.parse_san(san)
        except Exception:
            self._list_entry.configure(bg="#45232e")
            self._root.after(400, lambda: self._list_entry.configure(bg="#313244"))
            return
        raw_idx = self._view_idx - 1   # history index 1 → raw_moves[0]
        self.result = (raw_idx, san)
        self._root.destroy()

    def _abort(self):
        self.result = None
        self._root.destroy()


# ── Final review dialog ───────────────────────────────────────────────────────

class FinalReviewDialog:
    """
    Shows the completed game for review: navigable board, full move list,
    and the PGN text. Lets the user save to file and/or print before closing.
    """

    BG = "#1e1e2e"
    FG = "#cdd6f4"

    def __init__(self, game: chess.pgn.Game, pgn: str, default_save_path: str | None):
        self._pgn  = pgn
        self._save_path = default_save_path

        # Build position history from the game
        self._history: list[tuple[chess.Board, chess.Move | None, str]] = []
        b = chess.Board()
        self._history.append((b.copy(), None, "Start"))
        node = game
        while node.variations:
            node = node.variations[0]
            move = node.move
            san  = b.san(move)
            b.push(move)
            self._history.append((b.copy(), move, san))

        self._view_idx = len(self._history) - 1
        self._zoom = 1.0
        self._run()

    def _run(self):
        root = tk.Tk()
        root.title("Game Review")
        root.configure(bg=self.BG)
        root.resizable(True, True)
        self._root = root
        self._build(root)

        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h   = root.winfo_width(), root.winfo_height()
        root.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")
        root.mainloop()

    def _build(self, root):
        pad = dict(padx=10, pady=6)

        # Header
        hdr = tk.Frame(root, bg=self.BG)
        hdr.pack(fill=tk.X, **pad)
        tk.Label(hdr, text="Game complete — review before saving",
                 font=("Arial", 14, "bold"), fg="#a6e3a1", bg=self.BG).pack(anchor="w")
        tk.Label(hdr, text=f"{len(self._history)-1} moves",
                 font=("Arial", 11), fg="#a6adc8", bg=self.BG).pack(anchor="w")

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=10)

        # Main area: board+nav  |  move list  |  PGN text
        main = tk.Frame(root, bg=self.BG)
        main.pack(fill=tk.BOTH, expand=True, **pad)

        # — Board + nav —
        board_lf = tk.LabelFrame(main, text=" Position ",
                                  font=("Arial", 10, "bold"),
                                  fg=self.FG, bg=self.BG, bd=1, relief="groove")
        board_lf.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))

        self._board_canvas = BoardCanvas(board_lf,
                                          self._history[self._view_idx][0],
                                          last_move=self._history[self._view_idx][1],
                                          bg=self.BG)
        self._board_canvas.pack(padx=6, pady=6)

        nav = tk.Frame(board_lf, bg=self.BG)
        nav.pack(fill=tk.X, padx=6, pady=(0, 4))
        for txt, cmd in [("◀◀", self._go_start), ("◀", self._go_prev),
                         ("▶", self._go_next),   ("▶▶", self._go_end)]:
            tk.Button(nav, text=txt, font=("Arial", 11), width=3,
                      bg="#45475a", fg=self.FG, relief="flat", cursor="hand2",
                      command=cmd).pack(side=tk.LEFT, padx=1)
        self._pos_label = tk.Label(nav, text="", font=("Arial", 9),
                                    fg="#a6adc8", bg=self.BG, width=14)
        self._pos_label.pack(side=tk.LEFT, padx=6)

        root.bind("<Left>",  lambda _: self._go_prev())
        root.bind("<Right>", lambda _: self._go_next())
        root.bind("<Home>",  lambda _: self._go_start())
        root.bind("<End>",   lambda _: self._go_end())

        # — Move list —
        ml_lf = tk.LabelFrame(main, text=" Moves ",
                               font=("Arial", 10, "bold"),
                               fg=self.FG, bg=self.BG, bd=1, relief="groove")
        ml_lf.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))

        ml_scroll = tk.Scrollbar(ml_lf, orient="vertical")
        self._move_list = tk.Listbox(ml_lf, font=("Courier", 11),
                                     bg="#181825", fg=self.FG,
                                     selectbackground="#585b70",
                                     selectforeground="white",
                                     activestyle="none",
                                     width=14, height=16,
                                     yscrollcommand=ml_scroll.set,
                                     exportselection=False)
        ml_scroll.config(command=self._move_list.yview)
        ml_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._move_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._move_list.bind("<<ListboxSelect>>", self._on_list_select)

        self._move_list.insert(tk.END, "  Start")
        for i, (_, _, san) in enumerate(self._history[1:], start=1):
            num = (i + 1) // 2
            dot = "." if i % 2 == 1 else "…"
            self._move_list.insert(tk.END, f"  {num}{dot} {san}")

        # — PGN text —
        pgn_lf = tk.LabelFrame(main, text=" PGN ",
                                font=("Arial", 10, "bold"),
                                fg=self.FG, bg=self.BG, bd=1, relief="groove")
        pgn_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        pgn_scroll = tk.Scrollbar(pgn_lf, orient="vertical")
        self._pgn_text = tk.Text(pgn_lf, font=("Courier", 10),
                                  bg="#181825", fg=self.FG,
                                  insertbackground=self.FG,
                                  wrap=tk.WORD, width=36, height=16,
                                  yscrollcommand=pgn_scroll.set)
        pgn_scroll.config(command=self._pgn_text.yview)
        pgn_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._pgn_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._pgn_text.insert("1.0", self._pgn)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=10)

        # — Bottom bar: save path + buttons —
        bot = tk.Frame(root, bg=self.BG)
        bot.pack(fill=tk.X, **pad)

        tk.Label(bot, text="Save as:", font=("Arial", 11), fg=self.FG, bg=self.BG
                 ).pack(side=tk.LEFT)

        self._path_var = tk.StringVar(value=self._save_path or "game.pgn")
        tk.Entry(bot, textvariable=self._path_var,
                 font=("Arial", 11), width=28,
                 bg="#313244", fg=self.FG, insertbackground=self.FG,
                 relief="flat", bd=4).pack(side=tk.LEFT, padx=6)

        tk.Button(bot, text="💾  Save",
                  font=("Arial", 11, "bold"), bg="#a6e3a1", fg="#1e1e2e",
                  relief="flat", padx=12, pady=4, cursor="hand2",
                  command=self._save).pack(side=tk.LEFT, padx=4)

        tk.Button(bot, text="🖨  Print",
                  font=("Arial", 11), bg="#89b4fa", fg="#1e1e2e",
                  relief="flat", padx=12, pady=4, cursor="hand2",
                  command=self._print).pack(side=tk.LEFT, padx=4)

        tk.Button(bot, text="Done",
                  font=("Arial", 11), bg="#45475a", fg=self.FG,
                  relief="flat", padx=12, pady=4, cursor="hand2",
                  command=self._root.destroy).pack(side=tk.RIGHT)

        self._status = tk.Label(bot, text="", font=("Arial", 10),
                                 fg="#a6e3a1", bg=self.BG)
        self._status.pack(side=tk.RIGHT, padx=10)

        self._refresh_board_view()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _refresh_board_view(self):
        snap, last_move, san = self._history[self._view_idx]
        self._board_canvas.board = snap
        self._board_canvas.redraw(last_move)

        label = "Start" if self._view_idx == 0 else (
            f"{(self._view_idx+1)//2}{'.' if self._view_idx%2==1 else '…'} {san}")
        self._pos_label.configure(text=label)

        self._move_list.selection_clear(0, tk.END)
        self._move_list.selection_set(self._view_idx)
        self._move_list.see(self._view_idx)

    def _go_prev(self):
        if self._view_idx > 0:
            self._view_idx -= 1
            self._refresh_board_view()

    def _go_next(self):
        if self._view_idx < len(self._history) - 1:
            self._view_idx += 1
            self._refresh_board_view()

    def _go_start(self):
        self._view_idx = 0
        self._refresh_board_view()

    def _go_end(self):
        self._view_idx = len(self._history) - 1
        self._refresh_board_view()

    def _on_list_select(self, _event):
        sel = self._move_list.curselection()
        if sel:
            self._view_idx = sel[0]
            self._refresh_board_view()

    # ── Save / print ──────────────────────────────────────────────────────────

    def _save(self):
        path = self._path_var.get().strip()
        if not path:
            return
        Path(path).write_text(self._pgn)
        self._status.configure(text=f"Saved to {path}")

    def _print(self):
        import tempfile, subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".pgn", mode="w",
                                          delete=False, prefix="game_")
        tmp.write(self._pgn)
        tmp.close()
        # macOS: open with TextEdit (or default handler), which supports printing
        subprocess.Popen(["open", "-a", "TextEdit", tmp.name])
        self._status.configure(text="Opened in TextEdit — use ⌘P to print")


# ── Main processing pipeline ──────────────────────────────────────────────────

def process_scoresheet(image_path: str, output_path: str | None = None) -> str | None:
    """
    Full pipeline:
      1. Claude reads moves from the image.
      2. Each move is validated with python-chess.
      3. Illegal moves open the correction GUI.
      4. Returns the final PGN string (and optionally writes to a file).
    """
    print(f"\nProcessing: {image_path}")
    print("─" * 50)
    print("Sending image to Claude for move extraction…")

    raw_moves = extract_moves_from_image(image_path)

    if not raw_moves:
        print("ERROR: No moves could be extracted from the image.")
        return None

    print(f"Claude extracted {len(raw_moves)} moves: {raw_moves}\n")

    raw_moves = list(raw_moves)   # make mutable for backtrack edits
    confirmed_sans: list[str] = []
    idx = 0
    aborted = False

    while idx < len(raw_moves):
        raw       = raw_moves[idx]
        color_str = "White" if idx % 2 == 0 else "Black"
        num       = idx // 2 + 1
        dot       = "." if idx % 2 == 0 else "…"
        label     = f"{num}{dot} ({color_str})"

        # Rebuild board to current position
        board = chess.Board()
        for s in confirmed_sans:
            board.push(board.parse_san(s))

        # ── Try to parse the move ─────────────────────────────────────────────
        parsed = None

        try:
            parsed = board.parse_san(raw)
        except Exception:
            pass

        if parsed is None:
            fixed = fix_ocr_errors(raw)
            if fixed != raw:
                try:
                    parsed = board.parse_san(fixed)
                    print(f"  Auto-fixed  {label}  '{raw}' → '{fixed}'")
                    raw = fixed
                except Exception:
                    pass

        # ── Interactive correction ────────────────────────────────────────────
        if parsed is None:
            print(f"\n  !! Illegal move  {label}  '{raw}'  — opening correction dialog…")
            suggestions = suggest_similar_moves(board, raw)
            dlg = CorrectionDialog(
                image_path=image_path,
                board=board,
                move_label=label,
                ocr_move=raw,
                suggestions=suggestions,
                current_raw_idx=idx,
            )
            if dlg.result is None:
                print("  Aborted by user.")
                aborted = True
                break

            corrected_idx, corrected_san = dlg.result

            if corrected_idx < idx:
                # User edited an earlier move — backtrack
                print(f"  Backtracking to move {corrected_idx + 1}, replacing '{raw_moves[corrected_idx]}' → '{corrected_san}'")
                raw_moves[corrected_idx] = corrected_san
                confirmed_sans = confirmed_sans[:corrected_idx]
                idx = corrected_idx
                continue

            raw = corrected_san
            raw_moves[idx] = raw
            print(f"  User corrected to: '{raw}'")
            try:
                parsed = board.parse_san(raw)
            except Exception as exc:
                print(f"  ERROR: still can't parse '{raw}': {exc}")
                aborted = True
                break

        san = board.san(parsed)
        board.push(parsed)
        confirmed_sans.append(san)
        print(f"  OK  {label}  {san}")
        idx += 1

    # ── Rebuild game from confirmed moves ─────────────────────────────────────
    game = chess.pgn.Game()
    node = game
    b    = chess.Board()
    for san in confirmed_sans:
        move = b.parse_san(san)
        b.push(move)
        node = node.add_variation(move)
    board = b

    # ── Build PGN ─────────────────────────────────────────────────────────────
    game.headers["Event"]  = "?"
    game.headers["Site"]   = "?"
    game.headers["Date"]   = date.today().strftime("%Y.%m.%d")
    game.headers["Round"]  = "?"
    game.headers["White"]  = "?"
    game.headers["Black"]  = "?"
    game.headers["Result"] = board.result() if board.is_game_over() else "*"

    pgn = str(game)

    print(f"\n{'─'*50}\nFinal PGN:\n{pgn}")

    FinalReviewDialog(game=game, pgn=pgn, default_save_path=output_path)

    return pgn


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chess_pgn_scanner.py <image_path> [output.pgn]")
        sys.exit(1)

    img  = sys.argv[1]
    out  = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(img).exists():
        print(f"ERROR: File not found: {img}")
        sys.exit(1)

    result = process_scoresheet(img, out)
    sys.exit(0 if result is not None else 1)
