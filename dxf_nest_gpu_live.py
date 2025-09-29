#!/usr/bin/env python3
# dxf_nest_gpu_live.py — Live HTML viewer + SSE controls (pause/resume/stop), optional CUDA/NumPy
# - GPU accel via PyTorch when available; fast CPU fallback via NumPy if found; else pure Python
# - Thickness grouping by filename prefix "<thickness><unit>-*.dxf" (e.g., 0.5in-*, 12mm-*)
# - Guaranteed spacing with safety pixels; no shared-line cutting unless enabled
# - Output: per-thickness nested .dxf; optional split per sheet
# - Standalone HTML UI written to the folder; opens automatically; Pause/Resume/Stop & Save
#
# Changelog (2025-09-29):
# - FIX: replaced all occurrences of `2*math*pi` with TWO_PI (2 * math.pi) to prevent TypeError
# - LOGGING: added rotating file + console logging with context dumps and structured messages
# - HARDENING: defensive guards and error logging around DXF parsing, SSE server, placement, and writers

# ================= Default Settings (overridable by CLI) =================
FOLDER = r"C:\Users\Jsudhakaran\OneDrive - GN Corporation Inc\Desktop\test\For waterjet cutting"

SHEET_W = 48.0
SHEET_H = 96.0
SHEET_MARGIN = 0.50         # visual + writer margin around the sheet
SHEET_GAP = 2.0             # gap between sheet frames in the written DXF

SPACING  = 0.125            # requested minimum gap between parts (drawing units)
JOIN_TOL = 0.005
ARC_CHORD_TOL = 0.01

FALLBACK_OPEN_AS_BBOX = True

ALLOW_ROTATE_90   = True
ALLOW_MIRROR      = False
USE_OBB_CANDIDATE = True

INSUNITS = 1  # 1=inches (DXF header), 4=mm

RECT_ALIGN_MODE = "prefer"  # "off" | "prefer" | "force"
RECT_ALIGN_TOL  = 1e-3

ALLOW_NEST_IN_HOLES = True
NEST_MODE = "bitmap"         # "bitmap" | "shelf"
PIXELS_PER_UNIT = 20

BITMAP_EVAL_WORKERS = None
BITMAP_DEVICE = None  # "cuda", "cuda:0", "cpu", etc.

SHUFFLE_TRIES = 5
SHUFFLE_SEED  = None

GROUP_BY_THICKNESS = False
THICKNESS_LABEL_UNITS = "auto"  # "auto"|"in"|"mm"
SPLIT_SHEETS = False
MERGE_LINES  = True         # shared-line cutting OFF by default (preserve gap)
ROTATION_STEP_DEG = 0.0

# gap safety: small pixel cushion to avoid accidental touching due to rasterization
SAFETY_PX = 2
MIN_SPACING_PIXELS = 4

# Gap enforcement (new)
ENFORCE_GAP = True          # toggleable from the UI; governs placement-time dilation and enables validator

# Live UI / server
UI_FILENAME = "nest_viewer.html"
HTTP_HOST   = "127.0.0.1"
HTTP_PORT   = 0  # 0 = auto-pick a free port
# ========================================================================

import os, math, re, sys, json, time, webbrowser, threading, traceback, datetime, logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple, Dict, Optional, Any
from random import Random
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import contextlib

# ---- numeric constants ----
TWO_PI = 2 * math.pi

# ---- logging setup (initialized later when FOLDER is known) ----
LOGGER = logging.getLogger("dxf_nest_live")
LOGGER.addHandler(logging.NullHandler())
_LOG_FILE_PATH = None

def _init_logging(folder: str):
    global _LOG_FILE_PATH
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    # Console
    if not any(isinstance(h, logging.StreamHandler) for h in LOGGER.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        LOGGER.addHandler(ch)
    # File (rotating)
    log_dir = folder if os.path.isdir(folder) else os.getcwd()
    _LOG_FILE_PATH = os.path.join(log_dir, "nest_log.txt")
    fh = RotatingFileHandler(_LOG_FILE_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    LOGGER.info("Logging initialized. File: %s", _LOG_FILE_PATH)

def _dump_context():
    try:
        LOGGER.info("Context: sheet=%sx%s, margin=%.4f, spacing=%.4f, mode=%s, ppu=%d, tries=%d, seed=%s, "
                    "mirror=%s, holes=%s, rect-align=%s, group-by-thickness=%s, split=%s, merge=%s, insunits=%s, "
                    "device_pref=%s, enforce_gap=%s",
                    SHEET_W, SHEET_H, SHEET_MARGIN, SPACING, NEST_MODE, PIXELS_PER_UNIT, SHUFFLE_TRIES, SHUFFLE_SEED,
                    ALLOW_MIRROR, ALLOW_NEST_IN_HOLES, RECT_ALIGN_MODE, GROUP_BY_THICKNESS, SPLIT_SHEETS, MERGE_LINES,
                    "in" if INSUNITS==1 else "mm", BITMAP_DEVICE, ENFORCE_GAP)
    except Exception:
        LOGGER.exception("Failed dumping context")

# Fallback sample folder shipped with script
_REPO_SAMPLE_FOLDER = os.path.join(os.path.dirname(__file__), "For waterjet cutting")
if os.path.isdir(_REPO_SAMPLE_FOLDER):
    FOLDER = _REPO_SAMPLE_FOLDER

if not BITMAP_EVAL_WORKERS:
    cpu_count = os.cpu_count() or 1
    BITMAP_EVAL_WORKERS = max(1, cpu_count)

IS_WINDOWS = (os.name == "nt")

# Runtime toggles exposed in the HTML UI (key, label, global attr, description)
_UI_TOGGLE_DEFS = [
    ("allow_mirror", "Allow mirror / flip parts", "ALLOW_MIRROR", "Permit mirrored copies when searching poses."),
    ("allow_rotate_90", "Allow automatic 90° rotations", "ALLOW_ROTATE_90", "Include a 90° rotation candidate."),
    ("use_obb", "Use OBB seeding", "USE_OBB_CANDIDATE", "Seed placement with oriented bounding box pose."),
    ("allow_holes", "Allow nesting inside holes", "ALLOW_NEST_IN_HOLES", "Permit parts to be placed within other part holes."),
    ("fallback_bbox", "Fallback open profiles to bounding boxes", "FALLBACK_OPEN_AS_BBOX", "Treat open DXF profiles as rectangles."),
    ("group_by_thickness", "Group by thickness labels", "GROUP_BY_THICKNESS", "Nest files grouped by detected thickness."),
    ("split_sheets", "Split sheets into separate DXFs", "SPLIT_SHEETS", "Write one DXF per finished sheet."),
    ("merge_lines", "Merge touching lines", "MERGE_LINES", "Combine collinear edges for shared cutting."),
    ("enforce_gap", "Enforce minimum gap", "ENFORCE_GAP", "Guarantee ≥ SPACING between parts; also validates after placement."),
]

def _ui_toggle_snapshot():
    snap = []
    g = globals()
    for key, label, attr, desc in _UI_TOGGLE_DEFS:
        snap.append({
            "key": key,
            "label": label,
            "value": bool(g.get(attr)),
            "description": desc,
        })
    return snap

def _apply_toggle_config(cfg: Dict[str, Any]):
    if not isinstance(cfg, dict):
        LOGGER.warning("Toggle config is not a dict: %r", type(cfg))
        return
    g = globals()
    for key, _label, attr, _desc in _UI_TOGGLE_DEFS:
        if key in cfg:
            g[attr] = bool(cfg[key])
    LOGGER.info("Applied toggles: %s", json.dumps(cfg, ensure_ascii=False))

# ---------- Tiny Win progress window (optional) ----------
if IS_WINDOWS:
    import ctypes
    user32  = ctypes.windll.user32
    gdi32   = ctypes.windll.gdi32
    kernel32= ctypes.windll.kernel32

    UINT = ctypes.c_uint; DWORD = ctypes.c_uint; INT = ctypes.c_int; LONG = ctypes.c_long
    ULONG_PTR = ctypes.c_size_t; LONG_PTR  = ctypes.c_ssize_t
    WPARAM = ULONG_PTR; LPARAM = LONG_PTR; LRESULT = LONG_PTR
    HWND = ctypes.c_void_p; HINSTANCE = ctypes.c_void_p; HICON = ctypes.c_void_p
    HCURSOR = ctypes.c_void_p; HBRUSH = ctypes.c_void_p; HMENU = ctypes.c_void_p
    LPCWSTR = ctypes.c_wchar_p

    WS_OVERLAPPEDWINDOW = 0x00CF0000; WS_VISIBLE = 0x10000000
    WS_CHILD = 0x40000000; WS_EX_TOPMOST = 0x00000008
    SW_SHOWNORMAL = 1; WM_DESTROY = 0x0002; PM_REMOVE = 0x0001
    SS_LEFT = 0x00000000; SS_NOPREFIX = 0x00000080; WHITE_BRUSH = 0

    class POINT(ctypes.Structure): _fields_=[("x", LONG), ("y", LONG)]
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, UINT, WPARAM, LPARAM)
    class WNDCLASS(ctypes.Structure):
        _fields_=[("style", UINT),("lpfnWndProc", WNDPROC),("cbClsExtra", INT),("cbWndExtra", INT),
                  ("hInstance", HINSTANCE),("hIcon", HICON),("hCursor", HCURSOR),
                  ("hbrBackground", HBRUSH),("lpszMenuName", LPCWSTR),("lpszClassName", LPCWSTR)]
    class MSG(ctypes.Structure):
        _fields_=[("hwnd", HWND),("message", UINT),("wParam", WPARAM),("lParam", LPARAM),
                  ("time", DWORD),("pt", POINT)]

    user32.DefWindowProcW.argtypes=[HWND, UINT, WPARAM, LPARAM]
    user32.DefWindowProcW.restype=LRESULT
    user32.RegisterClassW.argtypes=[ctypes.POINTER(WNDCLASS)]
    user32.CreateWindowExW.argtypes=[DWORD, LPCWSTR, LPCWSTR, DWORD, INT, INT, INT, INT, HWND, HMENU, HINSTANCE, ctypes.c_void_p]
    user32.CreateWindowExW.restype=HWND
    gdi32.GetStockObject.argtypes=[INT]; gdi32.GetStockObject.restype=HBRUSH
    DefWindowProcW = user32.DefWindowProcW

    class WinProgress:
        def __init__(self, title="Nesting DXF…", width=520, height=220):
            self.enabled=True; self.title=title; self.width=width; self.height=height
            self.hInstance = kernel32.GetModuleHandleW(None)
            self.hwnd = HWND(); self.hStatic = HWND(); self._wndproc=None
        def create(self):
            try:
                @WNDPROC
                def wndproc(hwnd, msg, wParam, lParam):
                    if msg==WM_DESTROY:
                        user32.PostQuitMessage(0); return LRESULT(0)
                    try: return DefWindowProcW(hwnd, msg, wParam, lParam)
                    except Exception:
                        return LRESULT(0)
                self._wndproc = wndproc
                cls = WNDCLASS(); cls.lpfnWndProc = self._wndproc; cls.hInstance=self.hInstance
                cls.hbrBackground = gdi32.GetStockObject(WHITE_BRUSH); cls.lpszClassName="PyNestProgress"
                with contextlib.suppress(Exception):
                    user32.RegisterClassW(ctypes.byref(cls))
                sw=user32.GetSystemMetrics(0); sh=user32.GetSystemMetrics(1)
                x=max(0,(sw-self.width)//2); y=max(0,(sh-self.height)//2)
                self.hwnd=user32.CreateWindowExW(0,"PyNestProgress",self.title,
                    WS_OVERLAPPEDWINDOW|WS_VISIBLE,x,y,self.width,self.height,None,None,self.hInstance,None)
                self.hStatic=user32.CreateWindowExW(0,"STATIC","Loading…",WS_CHILD|WS_VISIBLE|SS_LEFT|SS_NOPREFIX,
                    12,12,self.width-24,self.height-24,self.hwnd,None,self.hInstance,None)
                user32.ShowWindow(self.hwnd, SW_SHOWNORMAL); self.pump()
            except Exception:
                LOGGER.exception("WinProgress create failed")
                self.enabled=False
        def pump(self):
            if not self.enabled: return
            try:
                msg = MSG()
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    user32.TranslateMessage(ctypes.byref(msg)); user32.DispatchMessageW(ctypes.byref(msg))
            except Exception:
                pass
        def update(self, text: str):
            if not self.enabled: return
            try: user32.SetWindowTextW(self.hStatic, text); self.pump()
            except Exception:
                pass
        def close(self):
            if not self.enabled: return
            with contextlib.suppress(Exception):
                user32.DestroyWindow(self.hwnd); self.hwnd=HWND()
else:
    class WinProgress:
        def __init__(self,*_,**__): self.enabled=False
        def create(self): pass
        def update(self,_): pass
        def pump(self): pass
        def close(self): pass

# --------- simple report + helper log ---------
_report_lines: List[str] = []
def log(line: str):
    print(line)
    _report_lines.append(line)
    try:
        LOGGER.info(line)
    except Exception:
        pass

Point = Tuple[float,float]
Loop  = List[Point]
Seg   = Tuple[Point,Point]

# ---------- Torch (optional) ----------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception as _ex:
    torch = None; F = None; _TORCH_OK = False

class TorchMaskOps:
    def __init__(self, device: Optional[str] = None):
        if not torch: raise RuntimeError("PyTorch not available")
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
    def zeros(self, H: int, W: int): return torch.zeros((H,W), dtype=torch.uint8, device=self.device)
    def mask_to_tensor(self, mask_01):
        H=len(mask_01); W=len(mask_01[0]) if H else 0
        t=torch.empty((H,W), dtype=torch.uint8, device=self.device)
        for y in range(H):
            row = mask_01[y]
            if isinstance(row,(bytearray,bytes)): row=list(row)
            t[y,:W]=torch.tensor(row, dtype=torch.uint8, device=self.device)
        return t
    def _disk_kernel(self, r:int):
        if r<=0: return torch.ones((1,1,1,1), dtype=torch.uint8, device=self.device)
        xs,ys=torch.meshgrid(torch.arange(-r,r+1,device=self.device), torch.arange(-r,r+1,device=self.device), indexing='ij')
        k=((xs*xs+ys*ys)<=(r*r)).to(torch.uint8)
        return k.view(1,1,k.shape[0],k.shape[1])
    def find_first_fit(self, occ_safe, test_mask_tensor):
        H,W=occ_safe.shape; ph,pw=test_mask_tensor.shape
        if H<ph or W<pw: return None
        x=occ_safe.unsqueeze(0).unsqueeze(0).to(torch.float32)
        k=test_mask_tensor.flip(0,1).unsqueeze(0).unsqueeze(0).to(torch.float32)
        heat=F.conv2d(x,k,stride=1); ok=(heat<=0.5)
        if not torch.any(ok): return None
        yy,xx=torch.where(ok[0,0]); y=int(yy.min().item()); x=int(xx[yy==y].min().item()); return (x,y)
    def or_mask(self, occ, raw_mask, ox:int, oy:int):
        ph,pw=raw_mask.shape; occ[oy:oy+ph,ox:ox+pw]|=raw_mask
    def or_dilated(self, occ, raw_or_shell, ox:int, oy:int, r:int):
        ph,pw=raw_or_shell.shape
        tile=torch.zeros_like(occ); tile[oy:oy+ph, ox:ox+pw]=raw_or_shell
        if r>0:
            k=self._disk_kernel(r).to(torch.float32)
            y=(F.conv2d(tile.unsqueeze(0).unsqueeze(0).to(torch.float32),k,padding=r)>0).to(torch.uint8)
            tile=y[0,0]
        occ|=tile
    def count_true(self, occ)->int: return int(occ.sum().item())

# ---------- NumPy (optional) ----------
try:
    import numpy as np
    _NUMPY_OK = True
except Exception:
    np = None
    _NUMPY_OK = False

class NumpyMaskOps:
    """CPU acceleration using NumPy (no GPU required)."""
    def __init__(self):
        if np is None:
            raise RuntimeError("NumPy not available")
        self.device = "numpy"
    def zeros(self, H: int, W: int):
        return np.zeros((H, W), dtype=np.uint8)
    def mask_to_tensor(self, mask_01):
        H = len(mask_01); W = len(mask_01[0]) if H else 0
        arr = np.zeros((H, W), dtype=np.uint8)
        for y in range(H):
            row = mask_01[y]
            if isinstance(row, (bytes, bytearray)):
                arr[y, :W] = np.frombuffer(row, dtype=np.uint8, count=W)
            else:
                arr[y, :W] = np.asarray(row, dtype=np.uint8)
        return arr
    def find_first_fit(self, occ, test_mask):
        """Find first (x,y) where (occ AND test)==0 using FFT correlation."""
        H, W = occ.shape
        ph, pw = test_mask.shape
        if H < ph or W < pw:
            return None
        shape = (H + ph - 1, W + pw - 1)
        f1 = np.fft.rfftn(occ.astype(np.float32), shape)
        f2 = np.fft.rfftn(np.flipud(np.fliplr(test_mask.astype(np.float32))), shape)
        heat_full = np.fft.irfftn(f1 * f2, shape)
        valid = heat_full[ph - 1:H, pw - 1:W]
        ok = valid <= 0.5
        if not np.any(ok):
            return None
        yy, xx = np.where(ok)
        y = int(yy.min())
        x = int(xx[yy == y].min())
        return (x, y)
    def or_mask(self, occ, raw_mask, ox: int, oy: int):
        ph, pw = raw_mask.shape
        occ[oy:oy + ph, ox:ox + pw] |= raw_mask
    def or_dilated(self, occ, raw_or_shell, ox: int, oy: int, r: int):
        ph, pw = raw_or_shell.shape
        tile = np.zeros_like(occ, dtype=np.uint8)
        tile[oy:oy + ph, ox:ox + pw] = raw_or_shell
        if r <= 0:
            occ |= tile
            return
        H, W = occ.shape
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy > r*r:
                    continue
                y0 = max(0, dy); y1 = min(H, H + dy)
                x0 = max(0, dx); x1 = min(W, W + dx)
                if y1 <= y0 or x1 <= x0:
                    continue
                occ[y0:y1, x0:x1] |= tile[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    def count_true(self, occ) -> int:
        return int(occ.sum())

def build_mask_ops(device_pref: Optional[str]):
    # Prefer PyTorch (CUDA or CPU). Else fallback to NumPy. Else None.
    if torch is not None:
        try:
            dev = device_pref.strip() if (device_pref and isinstance(device_pref, str)) else None
            if dev is None:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            _ = torch.zeros(1, device=torch.device(dev))
            LOGGER.info("Using PyTorch device: %s (cuda_available=%s)", dev, torch.cuda.is_available())
            return TorchMaskOps(dev)
        except Exception:
            LOGGER.exception("PyTorch device init failed; will try NumPy")
    if np is not None:
        try:
            LOGGER.info("Using NumPy acceleration (CPU)")
            return NumpyMaskOps()
        except Exception:
            LOGGER.exception("NumPy init failed")
    LOGGER.warning("No acceleration backend available; using pure Python paths")
    return None

# ---------- thickness + qty ----------
def _read_text(p):
    with open(p,'r',encoding='utf-8',errors='ignore') as f:
        return f.read().splitlines()

def _normalize_thickness_label(value: float, unit: str) -> str:
    s = f"{value:.4f}".rstrip('0').rstrip('.')
    return f"{s}{'in' if unit=='in' else 'mm'}"

def _parse_thickness_from_token(token: str, default_unit: str) -> Optional[Tuple[float,str]]:
    t = token.strip().lower()
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*(?:("|in(?:ch(?:es)?)?|mm|millimet(?:er|re)s?))$', t)
    if m:
        val=float(m.group(1)); raw_u=m.group(2); unit='in' if (raw_u=='"' or (raw_u and 'in' in raw_u)) else 'mm'
        return val, unit
    m = re.match(r'^([0-9]*\.?[0-9]+)$', t)
    if m: return float(m.group(1)), default_unit
    return None

def _parse_thickness_from_basename(basename: str, default_unit: str) -> Optional[Tuple[float,str]]:
    token = basename.split('-', 1)[0]
    return _parse_thickness_from_token(token, default_unit)

def _convert_thickness_for_label(value: float, unit: str, label_units: str) -> Tuple[float, str]:
    unit = unit.lower()
    if label_units == "auto": return value, unit
    if label_units == "in":   return ((value/25.4) if unit=="mm" else value, "in")
    if label_units == "mm":   return ((value*25.4) if unit in ("in", '"') else value, "mm")
    return value, unit

def read_qty_for_dxf(folder: str, dxf_filename: str) -> int:
    base, _ = os.path.splitext(dxf_filename)
    for ext in ('.txt', '.TXT'):
        p = os.path.join(folder, base + ext)
        if os.path.isfile(p):
            try:
                lines=[ln.strip() for ln in open(p,'r',encoding='utf-8',errors='ignore') if ln.strip()]
                if not lines: return 1
                start = 1 if ('quantity' in lines[0].lower()) else 0
                total=0
                for ln in lines[start:]:
                    cells=[c.strip() for c in ln.split(',')]
                    token=cells[-1] if cells else ln
                    q=None
                    try: q=int(float(token))
                    except:
                        digs=''
                        for ch in ln:
                            if ch.isdigit(): digs+=ch
                            elif digs: break
                        if digs: q=int(digs)
                    if q and q>0: total+=q
                return total if total>0 else 1
            except Exception:
                LOGGER.exception("Failed reading qty from: %s", p)
                return 1
    return 1

def read_thickness_label(folder: str, dxf_filename: str, label_units: str) -> str:
    default_unit = 'in' if INSUNITS == 1 else 'mm'
    base, _ = os.path.splitext(dxf_filename)
    parsed = _parse_thickness_from_basename(os.path.basename(base), default_unit)
    if parsed:
        v,u = parsed
        vv, uu = _convert_thickness_for_label(v,u,label_units)
        return _normalize_thickness_label(vv, uu)
    return "unknown"

# ---------- DXF parse/join ----------
def _arc_points(cx,cy,r,a0_deg,a1_deg,chord_tol):
    a0=math.radians(a0_deg); a1=math.radians(a1_deg)
    while a1<a0: a1+=TWO_PI
    sweep=a1-a0
    if r<=0: return [(cx,cy)]
    dtheta=2*math.asin(max(0.0,min(1.0,chord_tol/(2*r)))) if chord_tol>0 else (math.pi/36)
    steps=max(2,int(math.ceil(sweep/max(dtheta,1e-6))))
    return [(cx+r*math.cos(a0+sweep*k/steps), cy+r*math.sin(a0+sweep*k/steps)) for k in range(steps+1)]

def _ellipse_points(cx, cy, mx, my, ratio, t0, t1, chord_tol):
    maj_len = math.hypot(mx, my)
    if maj_len <= 0: return [(cx,cy)]
    vx, vy = (-my/maj_len), (mx/maj_len)
    a_vecx, a_vecy = mx, my
    b_len = maj_len * max(0.0, ratio)
    b_vecx, b_vecy = vx * b_len, vy * b_len
    while t1 < t0: t1 += TWO_PI
    sweep = t1 - t0
    steps = max(24, int(abs(sweep) / max(1e-6, 2*math.asin(min(1.0, chord_tol / max(1e-6, maj_len))))))  # cap step
    pts=[]
    for k in range(steps+1):
        t = t0 + sweep * (k/steps)
        x = cx + a_vecx*math.cos(t) + b_vecx*math.sin(t)
        y = cy + a_vecy*math.cos(t) + b_vecy*math.sin(t)
        pts.append((x,y))
    return pts

def parse_entities(path: str):
    try:
        lines=_read_text(path)
    except Exception:
        LOGGER.exception("Failed reading DXF text: %s", path)
        return [], []
    loops=[]; segs=[]
    in_entities=False
    in_lw=False; lw_pts=[]; lw_closed=False
    in_poly=False; poly_pts=[]; poly_closed=False
    in_spline=False; spline_fit=[]
    i=0; n=len(lines)
    def get(i): return lines[i].strip(), lines[i+1].strip()
    try:
        while i+1<n:
            code,val=get(i); i+=2
            if code=='0' and val=='SECTION':
                if i+1<n:
                    c2,v2=get(i)
                    if c2=='2' and v2=='ENTITIES': in_entities=True
                continue
            if code=='0' and val=='ENDSEC': in_entities=False; continue
            if not in_entities: continue
            if code=='0':
                # flush any open accumulators
                if in_lw:
                    if lw_pts:
                        if lw_closed and lw_pts[0]!=lw_pts[-1]: lw_pts.append(lw_pts[0])
                        if len(lw_pts)>=4: loops.append(lw_pts)
                    in_lw=False; lw_pts=[]; lw_closed=False
                if in_poly:
                    if poly_pts:
                        if poly_closed and poly_pts[0]!=poly_pts[-1]: poly_pts.append(poly_pts[0])
                        if len(poly_pts)>=4: loops.append(poly_pts)
                    in_poly=False; poly_pts=[]; poly_closed=False
                if in_spline:
                    if len(spline_fit)>=3:
                        pts = list(spline_fit)
                        if pts[0]!=pts[-1]: pts.append(pts[0])
                        if len(pts)>=4: loops.append(pts)
                    in_spline=False; spline_fit=[]
                if val=='LWPOLYLINE': in_lw=True; continue
                if val=='POLYLINE':   in_poly=True; continue
                if val=='SPLINE':     in_spline=True; spline_fit=[]; continue
                if val=='LINE':
                    x1=y1=x2=y2=None
                    while i+1<n:
                        c3,v3=get(i); i+=2
                        if c3=='0': i-=2; break
                        if c3=='10': x1=float(v3)
                        elif c3=='20': y1=float(v3)
                        elif c3=='11': x2=float(v3)
                        elif c3=='21': y2=float(v3)
                    if None not in (x1,y1,x2,y2): segs.append(((x1,y1),(x2,y2)))
                    continue
                if val in ('ARC','CIRCLE'):
                    cx=cy=r=None; a0=0.0 if val=='CIRCLE' else None; a1=360.0 if val=='CIRCLE' else None
                    while i+1<n:
                        c3,v3=get(i); i+=2
                        if c3=='0': i-=2; break
                        if   c3=='10': cx=float(v3)
                        elif c3=='20': cy=float(v3)
                        elif c3=='40': r =float(v3)
                        elif c3=='50': a0=float(v3)
                        elif c3=='51': a1=float(v3)
                    if None not in (cx,cy,r,a0,a1):
                        pts=_arc_points(cx,cy,r,a0,a1,ARC_CHORD_TOL)
                        for k in range(len(pts)-1):
                            segs.append((pts[k],pts[k+1]))
                    continue
                if val=='ELLIPSE':
                    cx=cy=mx=my=ratio=None; t0=0.0; t1=TWO_PI
                    while i+1<n:
                        c3,v3=get(i); i+=2
                        if c3=='0': i-=2; break
                        if c3=='10': cx=float(v3)
                        elif c3=='20': cy=float(v3)
                        elif c3=='11': mx=float(v3)
                        elif c3=='21': my=float(v3)
                        elif c3=='40': ratio=float(v3)
                        elif c3=='41': t0=float(v3)
                        elif c3=='42': t1=float(v3)
                    if None not in (cx,cy,mx,my,ratio):
                        pts=_ellipse_points(cx,cy,mx,my,ratio,t0,t1,ARC_CHORD_TOL)
                        for k in range(len(pts)-1):
                            segs.append((pts[k], pts[k+1]))
                    continue
                continue
            if in_lw:
                if code=='10':
                    x=float(val)
                    if i+1<n:
                        c2,v2=get(i); i+=2
                        if c2=='20':
                            lw_pts.append((x, float(v2)))
                        else:
                            i-=2
                elif code=='70':
                    try: flags=int(val)
                    except: flags=0
                    lw_closed=bool(flags&1)
            elif in_poly:
                if code=='70':
                    try: flags=int(val)
                    except: flags=0
                    poly_closed=bool(flags&1)
                elif code=='10':
                    x=float(val)
                    if i+1<n:
                        c2,v2=get(i); i+=2
                        if c2=='20':
                            poly_pts.append((x, float(v2)))
                        else:
                            i-=2
            elif in_spline:
                if code=='11':
                    x=float(val)
                    if i+1<n:
                        c2,v2=get(i); i+=2
                        if c2=='21':
                            spline_fit.append((x, float(v2)))
                        else:
                            i-=2
        if in_lw and lw_pts:
            if lw_closed and lw_pts[0]!=lw_pts[-1]: lw_pts.append(lw_pts[0])
            if len(lw_pts)>=4: loops.append(lw_pts)
        if in_poly and poly_pts:
            if poly_closed and poly_pts[0]!=poly_pts[-1]: poly_pts.append(poly_pts[0])
            if len(poly_pts)>=4: loops.append(poly_pts)
        if in_spline and len(spline_fit)>=3:
            pts=list(spline_fit)
            if pts[0]!=pts[-1]: pts.append(pts[0])
            if len(pts)>=4: loops.append(pts)
    except Exception:
        LOGGER.exception("DXF parse error at %s (near line %d)", path, i)
    return loops, segs

def join_segments_to_loops(segs: List[Seg], tol=JOIN_TOL) -> List[Loop]:
    if not segs: return []
    def key(pt): return (round(pt[0]/tol), round(pt[1]/tol))
    adj: Dict[tuple,List[tuple]]={}; used=[False]*len(segs)
    for idx,(a,b) in enumerate(segs):
        ka,kb=key(a),key(b)
        adj.setdefault(ka,[]).append((a,b,idx))
        adj.setdefault(kb,[]).append((b,a,idx))
    loops=[]
    for idx,(a0,b0) in enumerate(segs):
        if used[idx]: continue
        chain=[a0,b0]; used[idx]=True
        end=b0; kend=key(end)
        while True:
            nxt=None
            for a,b,j in adj.get(kend,[]):
                if used[j]: continue
                if abs(a[0]-end[0])<=tol and abs(a[1]-end[1])<=tol: nxt=(b,j); break
            if not nxt: break
            chain.append(nxt[0]); used[nxt[1]]=True; end=nxt[0]; kend=key(end)
        start=a0; kstart=key(start)
        while True:
            prv=None
            for a,b,j in adj.get(kstart,[]):
                if used[j]: continue
                if abs(b[0]-start[0])<=tol and abs(b[1]-start[1])<=tol: prv=(a,j); break
            if not prv: break
            chain.insert(0,prv[0]); used[prv[1]]=True; start=prv[0]; kstart=key(start)
        if len(chain)>=4 and abs(chain[0][0]-chain[-1][0])<=tol and abs(chain[0][1]-chain[-1][1])<=tol:
            if chain[0]!=chain[-1]: chain.append(chain[0])
            loops.append(chain)
    return loops

# ---------- geometry helpers ----------
def polygon_area(loop: Loop) -> float:
    s=0.0
    for i in range(len(loop)-1):
        x1,y1=loop[i]; x2,y2=loop[i+1]
        s += x1*y2 - x2*y1
    return 0.5*s

def bbox_of_points(pts: List[Tuple[float,float]]):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)

def bbox_of_loops(loops: List[Loop]):
    pts=[p for lp in loops for p in lp]
    return bbox_of_points(pts) if pts else (0,0,0,0)

def translate_loop(loop: Loop, dx: float, dy: float) -> Loop:
    return [(x+dx,y+dy) for x,y in loop]

def mirror_loop(loop: Loop) -> Loop:
    mirrored = [(-x, y) for x, y in loop]
    minx = min((x for x, _ in mirrored), default=0.0)
    miny = min((y for _, y in mirrored), default=0.0)
    return [(x - minx, y - miny) for x, y in mirrored]

def rotate_loop(loop: Loop, theta: float) -> Loop:
    c,s=math.cos(theta), math.sin(theta)
    rot=[(x*c - y*s, x*s + y*c) for x,y in loop]
    minx=min(x for x,_ in rot); miny=min(y for _,y in rot)
    return [(x-minx,y-miny) for x,y in rot]

def convex_hull(points: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    pts=sorted(set(points))
    if len(pts)<=1: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]; upper=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def min_area_rect(points: List[Tuple[float,float]]):
    hull=convex_hull(points)
    if len(hull)<=1: return 0.0,0.0,0.0
    best=(float('inf'),0.0,0.0,0.0)
    for i in range(len(hull)):
        x1,y1=hull[i]; x2,y2=hull[(i+1)%len(hull)]
        theta=math.atan2(y2-y1, x2-x1)
        ct,st=math.cos(-theta), math.sin(-theta)
        xs=[px*ct - py*st for px,py in hull]
        ys=[px*st + py*ct for px,py in hull]
        w=max(xs)-min(xs); h=max(ys)-min(ys); area=w*h
        if area<best[0]: best=(area,w,h,theta)
    _,w,h,theta=best
    return w,h,theta

def is_rect_like_by_area(outer_loop, obb_w, obb_h, tol_frac=RECT_ALIGN_TOL) -> bool:
    rect_area = obb_w * obb_h
    if rect_area <= 0: return False
    poly_area = abs(polygon_area(outer_loop))
    return abs(poly_area - rect_area) <= tol_frac * rect_area

def split_outer_and_holes(loops: List[Loop]):
    if not loops: return None,[]
    idx=max(range(len(loops)), key=lambda i: abs(polygon_area(loops[i])))
    return loops[idx], [loops[i] for i in range(len(loops)) if i!=idx]

# ---------- Part ----------
class Part:
    _uid_counter = 0
    def __init__(self, name: str, loops: List[Loop], fallback_bbox: Optional[Tuple[float,float,float,float]]):
        if loops:
            minx,miny,maxx,maxy=bbox_of_loops(loops)
            loops0=[translate_loop(lp,-minx,-miny) for lp in loops]
        elif fallback_bbox is not None:
            minx,miny,maxx,maxy=fallback_bbox
            loops0=[[ (0,0),(maxx-minx,0),(maxx-minx,maxy-miny),(0,maxy-miny),(0,0) ]]
        else:
            loops0=[]
        self.name=name
        if not loops0:
            self.outer=None; self.holes=[]; self.w=self.h=0.0; self.obb_w=self.obb_h=self.obb_theta=0.0; return
        self.outer,self.holes = split_outer_and_holes(loops0)
        minx,miny,maxx,maxy=bbox_of_loops([self.outer])
        self.w=maxx-minx; self.h=maxy-miny
        self.obb_w,self.obb_h,self.obb_theta = min_area_rect(self.outer)
        self._cand_cache: Dict[Tuple[Any, ...], Dict[str,Any]] = {}
        self.uid = Part._uid_counter; Part._uid_counter += 1

    def _axis_align_angles(self):
        a = (-self.obb_theta) % math.pi
        return [a, (a + math.pi/2) % math.pi]
    def is_rect_like(self) -> bool:
        return self.outer is not None and is_rect_like_by_area(self.outer, self.obb_w, self.obb_h)

    def candidate_angles(self):
        base = []
        if ROTATION_STEP_DEG and ROTATION_STEP_DEG > 0:
            step = math.radians(ROTATION_STEP_DEG)
            k = max(1, int(round(math.pi / step)))
            base = [(i*step) % math.pi for i in range(k)]
        else:
            base = [0.0]
            if ALLOW_ROTATE_90: base.append(math.pi/2)
            if USE_OBB_CANDIDATE and self.obb_w>0 and self.obb_h>0:
                a = self.obb_theta % math.pi
                base += [a, (a + math.pi/2) % math.pi]
        if RECT_ALIGN_MODE in ("prefer","force") and self.is_rect_like():
            axis = self._axis_align_angles()
            base = (axis if RECT_ALIGN_MODE=="force" else (axis + base))
        out=[]
        for a in base:
            if all(abs((a-b)%(math.pi))>math.radians(1) for b in out):
                out.append(a%(math.pi))
        return out

    def candidate_poses(self):
        angles = self.candidate_angles()
        mirrors = [False, True] if ALLOW_MIRROR else [False]
        seen=set(); poses=[]
        for mirror in mirrors:
            for ang in angles:
                key=(mirror, round((ang % TWO_PI),10))
                if key in seen: continue
                seen.add(key); poses.append((ang, mirror))
        return poses

    def oriented(self, theta: float, mirror: bool = False):
        if self.outer is None: return 0.0,0.0,[]
        loops_src = [self.outer] + self.holes
        if mirror: loops_src = [mirror_loop(lp) for lp in loops_src]
        if (abs(theta) % TWO_PI) > 1e-12:
            loops_src = [rotate_loop(lp, theta) for lp in loops_src]
        minx,miny,maxx,maxy=bbox_of_loops([loops_src[0]])
        return (maxx-minx),(maxy-miny),loops_src

# ---------- Candidate cache helpers ----------
def _candidate_cache_key(scale: int, angle: float, mirror: bool, spacing: float,
                         allow_holes: bool, enforce_gap: bool, safety_px: int) -> Tuple[Any, ...]:
    return (
        int(scale),
        round((angle % TWO_PI), 9),
        bool(mirror),
        round(float(spacing), 6),
        bool(allow_holes),
        bool(enforce_gap),
        int(safety_px),
    )


def _get_part_candidate(part: 'Part', scale: int, angle: float, mirror: bool, spacing: float,
                        allow_holes: bool, enforce_gap: bool) -> Dict[str, Any]:
    key = _candidate_cache_key(scale, angle, mirror, spacing, allow_holes, enforce_gap, SAFETY_PX)
    cand = part._cand_cache.get(key)
    if cand is None:
        w_units, h_units, loops = part.oriented(angle, mirror)
        raw, pw, ph = rasterize_loops(loops, scale)
        outer, _, _ = rasterize_outer_only(loops, scale)

        base = raw if allow_holes else outer
        occ_pad = dilate_mask(base, pw, ph, SAFETY_PX)
        spacing_px = int(math.ceil(spacing * scale)) if (enforce_gap and spacing > 0) else 0
        enforce_r = SAFETY_PX + spacing_px
        test = dilate_mask(base, pw, ph, enforce_r)
        test_segments, _ = _mask_segments_and_fills(test)
        raw_segments, raw_fills = _mask_segments_and_fills(raw)
        occ_segments, occ_fills = _mask_segments_and_fills(occ_pad)

        cand = {
            'loops': loops,
            'pw': pw,
            'ph': ph,

            'w_units': w_units,
            'h_units': h_units,
            'raw': raw,
            'raw_segments': raw_segments,
            'raw_fills': raw_fills,
            'occ': occ_pad,

            'occ_segments': occ_segments,
            'occ_fills': occ_fills,
            'test': test,
            'test_segments': test_segments,
            'tensor_cache': {},
        }
        part._cand_cache[key] = cand
    return cand


def _ensure_mask_tensors(cand: Dict[str, Any], mask_ops: Any) -> Dict[str, Any]:
    cache = cand.setdefault('tensor_cache', {})
    key = id(mask_ops)
    bundle = cache.get(key)
    if bundle is None:
        bundle = {
            'test': mask_ops.mask_to_tensor(cand['test']),
            'occ': mask_ops.mask_to_tensor(cand['occ']),
            'raw': mask_ops.mask_to_tensor(cand['raw']),
        }
        cache[key] = bundle
    return bundle

# ---------- Raster helpers ----------
def _empty_mask(w:int, h:int): return [bytearray(w) for _ in range(h)]

def _mask_segments_and_fills(mask):
    segments=[]; fills=[]
    for row in mask:
        row_segments=[]; row_fills=[]
        start=-1
        row_len=len(row)
        for idx,val in enumerate(row):
            if val:
                if start==-1:
                    start=idx
            elif start!=-1:
                if idx>start:
                    row_segments.append((start, idx))
                    row_fills.append(b"\x01"*(idx-start))
                start=-1
        if start!=-1 and row_len>start:
            row_segments.append((start, row_len))
            row_fills.append(b"\x01"*(row_len-start))
        segments.append(row_segments)
        fills.append(row_fills)
    return segments, fills

def pad_mask(mask, w, h, pad):
    pad = int(max(0, pad))
    if pad <= 0:
        return mask, w, h, 0, 0
    new_w = w + 2*pad
    new_h = h + 2*pad
    padded = [bytearray(new_w) for _ in range(new_h)]
    for y in range(h):
        src_row = mask[y]
        dst_row = padded[y + pad]
        dst_row[pad:pad + w] = src_row
    return padded, new_w, new_h, pad, pad

def rasterize_polygon_to_mask(mask, w, h, pts_scaled):
    if not pts_scaled: return
    ys=[p[1] for p in pts_scaled]
    y0=max(0,int(math.floor(min(ys)))); y1=min(h-1,int(math.ceil(max(ys))))
    n=len(pts_scaled)
    for y in range(y0,y1+1):
        yscan=y+0.5; xs=[]
        for i in range(n):
            x1,y1 = pts_scaled[i]; x2,y2 = pts_scaled[(i+1)%n]
            if y1==y2: continue
            if y1>y2: x1,y1,x2,y2=x2,y2,x1,y1
            if y1 <= yscan and yscan < y2:
                t=(yscan-y1)/(y2-y1); xs.append(x1+t*(x2-x1))
        if not xs: continue
        xs.sort()
        for i in range(0,len(xs),2):
            x_start=int(math.floor(xs[i])); x_end=int(math.ceil(xs[i+1]))-1 if i+1<len(xs) else x_start
            if x_end<0 or x_start>=w: continue
            x_start=max(0,x_start); x_end=min(w-1,x_end)
            row=mask[y]
            for x in range(x_start,x_end+1): row[x]=1

def rasterize_loops(loops: List[Loop], scale: float):
    allpts=[p for lp in loops for p in lp]
    if not allpts: return _empty_mask(1,1),1,1
    minx,miny,maxx,maxy=bbox_of_points(allpts)
    loops0=[[(x-minx,y-miny) for (x,y) in lp] for lp in loops]
    pw=max(1,int(math.ceil((maxx-minx)*scale))); ph=max(1,int(math.ceil((maxy-miny)*scale)))
    mask=_empty_mask(pw,ph)
    if loops0:
        outer=loops0[0]; outer_px=[(x*scale,y*scale) for x,y in outer]
        rasterize_polygon_to_mask(mask,pw,ph,outer_px)
        for hole in loops0[1:]:
            hole_px=[(x*scale,y*scale) for x,y in hole]
            hmask=_empty_mask(pw,ph); rasterize_polygon_to_mask(hmask,pw,ph,hole_px)
            for y in range(ph):
                row=mask[y]; hr=hmask[y]
                for x in range(pw):
                    if hr[x]: row[x]=0
    return mask,pw,ph

def rasterize_loops_with_bbox(loops: List[Loop], scale: float):
    """Like rasterize_loops but also returns original bbox minx,miny used for placement."""
    allpts=[p for lp in loops for p in lp]
    if not allpts: return _empty_mask(1,1),1,1,0.0,0.0
    minx,miny,maxx,maxy=bbox_of_points(allpts)
    loops0=[[(x-minx,y-miny) for (x,y) in lp] for lp in loops]
    pw=max(1,int(math.ceil((maxx-minx)*scale))); ph=max(1,int(math.ceil((maxy-miny)*scale)))
    mask=_empty_mask(pw,ph)
    if loops0:
        outer=loops0[0]; outer_px=[(x*scale,y*scale) for x,y in outer]
        rasterize_polygon_to_mask(mask,pw,ph,outer_px)
        for hole in loops0[1:]:
            hole_px=[(x*scale,y*scale) for x,y in hole]
            hmask=_empty_mask(pw,ph); rasterize_polygon_to_mask(hmask,pw,ph,hole_px)
            for y in range(ph):
                row=mask[y]; hr=hmask[y]
                for x in range(pw):
                    if hr[x]: row[x]=0
    return mask,pw,ph,minx,miny

def dilate_mask(mask,w,h,r):
    if r<=0: return mask
    out=_empty_mask(w,h); offs=[]; rr=r*r
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if dx*dx+dy*dy<=rr: offs.append((dx,dy))
    for y in range(h):
        row=mask[y]
        for x in range(w):
            if row[x]:
                for dx,dy in offs:
                    xx=x+dx; yy=y+dy
                    if 0<=xx<w and 0<=yy<h: out[yy][xx]=1
    return out

def _eff_scale(scale:int, spacing:float)->int:
    if spacing<=0: return scale
    return max(scale, int(math.ceil(MIN_SPACING_PIXELS/max(spacing,1e-9))))

# === outer-only raster (used when forbidding hole nesting) ===
def rasterize_outer_only(loops: List[Loop], scale: float):
    """Rasterize only the outer loop (no hole subtraction)."""
    if not loops:
        return _empty_mask(1,1), 1, 1
    outer = loops[0]
    if not outer:
        return _empty_mask(1,1), 1, 1
    xs=[x for x,_ in outer]; ys=[y for _,y in outer]
    minx,miny,maxx,maxy=min(xs),min(ys),max(xs),max(ys)
    outer0=[(x-minx, y-miny) for x,y in outer]
    pw=max(1,int(math.ceil((maxx-minx)*scale))); ph=max(1,int(math.ceil((maxy-miny)*scale)))
    mask=_empty_mask(pw,ph)
    outer_px=[(x*scale,y*scale) for x,y in outer0]
    rasterize_polygon_to_mask(mask,pw,ph,outer_px)
    return mask,pw,ph

# ---------- Control / SSE Server ----------
class NestAbortPartial(Exception):
    def __init__(self, placements, sheets_used): super().__init__("Stopped by user"); self.placements=placements; self.sheets=sheets_used

class NestControl:
    def __init__(self):
        self.pause = threading.Event()
        self.stop  = threading.Event()
        self.status_lock = threading.Lock()
        self.status = {"phase":"idle"}
        self._start_lock = threading.Lock()
        self._start_event = threading.Event()
        self._start_payload: Dict[str, Any] = {}
        self._started = False
    def set_status(self, **kv):
        with self.status_lock:
            self.status.update(kv)
    def get_status(self):
        with self.status_lock:
            return dict(self.status)
    def request_start(self, payload: Dict[str, Any]):
        with self._start_lock:
            if self._started:
                return False
            self._start_payload = dict(payload or {})
            self._started = True
        self.stop.clear()
        self.pause.clear()
        self.set_status(phase="starting")
        self._start_event.set()
        return True
    def wait_for_start(self) -> Dict[str, Any]:
        self._start_event.wait()
        with self._start_lock:
            return dict(self._start_payload)

class SSEHub:
    def __init__(self): self._clients=[]; self._lock=threading.Lock()
    def attach(self, handler):
        with self._lock: self._clients.append(handler)
    def detach(self, handler):
        with self._lock:
            if handler in self._clients: self._clients.remove(handler)
    def broadcast(self, typ:str, payload:dict):
        data=json.dumps({"type":typ, **payload})
        dead=[]
        with self._lock:
            for h in self._clients:
                try:
                    h.wfile.write(b"data: "); h.wfile.write(data.encode("utf-8")); h.wfile.write(b"\n\n")
                    h.wfile.flush()
                except Exception:
                    dead.append(h)
            for h in dead:
                try: self._clients.remove(h)
                except: pass

def write_standalone_html(path_on_disk: str):
    # Use doubled braces inside f-string for CSS/JS blocks.
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>DXF Nesting — Live</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root {{ --bg:#0b0f14; --fg:#e8eef6; --muted:#9fb3c8; --accent:#5ee1a2; --danger:#ff6b6b; --warn:#ffbb33; }}
  html,body {{ margin:0; height:100%; background:var(--bg); color:var(--fg); font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial; }}
  header {{ padding:12px 16px; display:flex; gap:12px; align-items:center; border-bottom:1px solid #223; }}
  .title {{ font-weight:700; letter-spacing:.2px; }}
  .spacer {{ flex:1; }}
  button {{ background:#1b2735; color:var(--fg); border:1px solid #2b3b52; border-radius:10px; padding:8px 12px; cursor:pointer; }}
  button:hover {{ border-color:#3f5b7a; }}
  button:active {{ transform: translateY(1px); }}
  button[disabled] {{ opacity:0.5; cursor:not-allowed; }}
  button.danger {{ border-color:var(--danger); color:var(--danger); }}
  button.accent {{ border-color:var(--accent); color:var(--accent); }}
  main {{ display:grid; grid-template-columns: 320px 1fr; min-height: calc(100% - 60px); }}
  aside {{ padding:14px; border-right:1px solid #223; }}
  section.viewer {{ position:relative; }}
  #canvasWrap {{ position:absolute; inset:0; display:flex; }}
  canvas {{ margin:auto; background:#0f1620; border:1px solid #223; border-radius:12px; }}
  .row {{ margin-bottom:12px; }}
  .label {{ color:var(--muted); font-size:12px; margin-bottom:4px; }}
  #progress {{ width:100%; height:14px; border-radius:8px; background:#121a24; overflow:hidden; border:1px solid #223; }}
  #bar {{ height:100%; width:0%; background:linear-gradient(90deg,var(--accent),#23a8f2); }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  select, input[type=number] {{ background:#0f1620; color:var(--fg); border:1px solid #223; border-radius:8px; padding:6px; width:100%; }}
  .small {{ font-size:12px; color:var(--muted); }}
  .pill {{ font-size:12px; padding:2px 8px; border:1px solid #2b3b52; border-radius:999px; }}
  .ok {{ border-color:var(--accent); color:var(--accent); }}
  .warn {{ border-color:var(--warn); color:var(--warn); }}
  #optionsWrap {{ display:flex; flex-direction:column; gap:6px; margin:8px 0 10px; }}
  .optCheck {{ display:flex; align-items:center; gap:8px; font-size:13px; }}
  .optCheck input {{ width:16px; height:16px; }}
</style>
</head>
<body>
<header>
  <div class="title">DXF Nesting — Live Viewer</div>
  <div class="pill" id="gpuPill">GPU: ?</div>
  <div class="pill" id="groupPill">Group: —</div>
  <div class="pill" id="sheetPill">Sheet: — / —</div>
  <div class

  <div class="spacer"></div>
  <button id="pauseBtn" disabled>Pause</button>
  <button id="resumeBtn" class="accent" disabled>Resume</button>
  <button id="stopBtn" class="danger" disabled>Stop &amp; Save</button>
</header>
<main>
  <aside>
    <div class="row" id="configPanel">
      <div class="label">Run setup</div>
      <div class="small" id="configMsg">Loading options…</div>
      <div id="optionsWrap"></div>
      <button id="startBtn" class="accent" disabled>Start Nesting</button>
    </div>
    <div class="row">
      <div class="label">Progress</div>
      <div id="progress"><div id="bar"></div></div>
      <div class="small"><span id="placed">0</span> / <span id="total">0</span> parts placed</div>
    </div>
    <div class="row">
      <div class="label">Status</div>
      <div class="mono" id="status">—</div>
    </div>
    <div class="row">
      <div class="label">View sheet</div>
      <select id="sheetSelect"></select>
    </div>
    <div class="row">
      <div class="label">Notes</div>
      <div class="small" id="notes">Gap enforcement uses your SPACING; validator double-checks after placement.</div>
    </div>
    <div class="row">
      <div class="label">Outputs</div>
      <div class="small mono" id="outputs">—</div>
    </div>
  </aside>
  <section class="viewer">
    <div id="canvasWrap"><canvas id="cv" width="1280" height="720"></canvas></div>
  </section>
</main>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let W=48, H=96, M=0.5, scale=1;
let currentSheet = 0;
let total = 0, placed = 0;
const pauseBtn = document.getElementById('pauseBtn');
const resumeBtn = document.getElementById('resumeBtn');
const stopBtn = document.getElementById('stopBtn');
const startBtn = document.getElementById('startBtn');
const optionsWrap = document.getElementById('optionsWrap');
const configMsg = document.getElementById('configMsg');
const sheetSel = document.getElementById('sheetSelect');

let runStarted = false;
let startRequested = false;

// Maintain sheet → paths in a Map
let sheets = new Map(); // sheetIndex -> {{paths: [ [ [x,y],... ] ], bbox:[W,H], margin:M}}

function setRunState(active) {{
  runStarted = active;
  [pauseBtn, resumeBtn, stopBtn].forEach(btn => {{ if (btn) btn.disabled = !active; }});
  optionsWrap.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.disabled = active; }});
  if (active) {{
    startBtn.disabled = true;
    startBtn.textContent = 'Running…';
  }}
}}

function renderOptions(opts) {{
  optionsWrap.innerHTML = '';
  if (!Array.isArray(opts) || opts.length === 0) {{
    const msg = document.createElement('div');
    msg.className = 'small';
    msg.textContent = 'No runtime toggles exposed.';
    optionsWrap.appendChild(msg);
    return;
  }}
  for (const opt of opts) {{
    const id = `opt_${{opt.key}}`;
    const label = document.createElement('label');
    label.className = 'optCheck';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = id;
    cb.dataset.key = opt.key;
    cb.checked = !!opt.value;
    cb.disabled = runStarted || startRequested;
    const span = document.createElement('span');
    span.textContent = opt.label;
    if (opt.description) span.title = opt.description;
    label.appendChild(cb);
    label.appendChild(span);
    optionsWrap.appendChild(label);
  }}
}}

async function fetchConfig() {{
  try {{
    const resp = await fetch('/config');
    if (!resp.ok) throw new Error('HTTP '+resp.status);
    const data = await resp.json();
    if (Array.isArray(data.options)) renderOptions(data.options);
    const phase = data.status && data.status.phase;
    if (!runStarted && (phase === 'waiting' || phase === 'idle')) {{
      configMsg.textContent = 'Adjust options then press Start.';
      startBtn.disabled = false;
      startBtn.textContent = 'Start Nesting';
    }}
  }} catch (err) {{
    configMsg.textContent = 'Failed to load options.';
  }}
}}

startBtn.addEventListener('click', async () => {{
  if (runStarted) return;
  const payload = {{}};
  optionsWrap.querySelectorAll('input[type=checkbox]').forEach(cb => {{
    payload[cb.dataset.key] = cb.checked;
  }});
  startRequested = true;
  optionsWrap.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.disabled = true; }});
  startBtn.disabled = true;
  startBtn.textContent = 'Starting…';
  configMsg.textContent = 'Submitting…';
  try {{
    const res = await fetch('/control?cmd=start', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload)
    }});
    if (!res.ok) {{
      const text = await res.text();
      throw new Error(text || ('HTTP '+res.status));
    }}
    configMsg.textContent = 'Waiting for backend…';
  }} catch (err) {{
    startRequested = false;
    optionsWrap.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.disabled = false; }});
    startBtn.disabled = false;
    startBtn.textContent = 'Start Nesting';
    configMsg.textContent = 'Start failed: '+err.message;
  }}
}});

pauseBtn.onclick = () => fetch('/control?cmd=pause', {{method:'POST'}});
resumeBtn.onclick = () => fetch('/control?cmd=resume', {{method:'POST'}});
stopBtn.onclick = () => fetch('/control?cmd=stop', {{method:'POST'}});

setRunState(false);
fetchConfig();

function fitScale() {{
  const pad = 20;
  const availW = cv.width - pad*2;
  const availH = cv.height - pad*2;
  const drawW = W + 2*M; const drawH = H + 2*M;
  scale = Math.min(availW/drawW, availH/drawH);
}}
function toXY(x,y) {{
  // origin bottom-left visual (flip Y)
  const pad = 20;
  const X = pad + (x)*scale;
  const Y = cv.height - (pad + (y)*scale);
  return [X,Y];
}}
function redraw() {{
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle = '#0f1620'; ctx.fillRect(0,0,cv.width,cv.height);
  fitScale();
  // frame
  ctx.strokeStyle='#445'; ctx.lineWidth=2;
  let [x0,y0] = toXY(0,0);
  let [x1,y1] = toXY(W+2*M,0);
  let [x2,y2] = toXY(W+2*M,H+2*M);
  let [x3,y3] = toXY(0,H+2*M);
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x3,y3); ctx.closePath(); ctx.stroke();
  // parts on current sheet
  const data = sheets.get(currentSheet);
  if (!data) return;
  ctx.strokeStyle='#5ee1a2'; ctx.lineWidth=1;
  for (const lp of data.paths) {{
    ctx.beginPath();
    for (let i=0;i<lp.length;i++) {{
      const [X,Y] = toXY(M+lp[i][0], M+lp[i][1]);
      if (i===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
    }}
    ctx.stroke();
  }}
}}
function setProgress(pPlaced, pTotal) {{
  placed=pPlaced; total=pTotal;
  document.getElementById('placed').textContent=placed;
  document.getElementById('total').textContent=pTotal;
  const pct = pTotal>0 ? (100*placed/pTotal) : 0;
  document.getElementById('bar').style.width = pct.toFixed(1)+'%';
}}
function ensureSheet(i) {{
  if (!sheets.has(i)) sheets.set(i, {{paths:[], bbox:[W,H], margin:M}});
  if (![...sheetSel.options].some(o => Number(o.value)===i)) {{
    const opt=document.createElement('option'); opt.value=String(i); opt.textContent='Sheet '+(i+1); sheetSel.appendChild(opt);
  }}
}}
sheetSel.addEventListener('change', () => {{ currentSheet = Number(sheetSel.value)||0; redraw(); }});

function setStatus(t) {{ document.getElementById('status').textContent = t; }}
function setGPU(ok) {{
  const pill=document.getElementById('gpuPill');
  pill.textContent='GPU: '+(ok?'ON':'OFF');
  pill.className = 'pill '+(ok?'ok':'warn');
}}
function setGroup(g) {{ const el=document.getElementById('groupPill'); el.textContent='Group: '+(g||'—'); }}
function setSheetPill(cur, total) {{ const el=document.getElementById('sheetPill'); el.textContent='Sheet: '+(cur+1)+' / '+(total||'—'); }}

const es = new EventSource('/events');
es.onmessage = (ev) => {{
  try {{
    const msg = JSON.parse(ev.data||'{{}}');
    if (msg.type==='waiting') {{
      setRunState(false);
      startRequested = false;
      if (Array.isArray(msg.options)) renderOptions(msg.options);
      const text = msg.message || 'Waiting for Start…';
      configMsg.textContent = text;
      startBtn.disabled = false;
      startBtn.textContent = 'Start Nesting';
      setStatus(text);
      return;
    }}
    if (msg.type==='starting') {{
      startRequested = true;
      optionsWrap.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.disabled = true; }});
      configMsg.textContent = 'Starting…';
      startBtn.disabled = true;
      startBtn.textContent = 'Starting…';
      setStatus('Starting…');
      return;
    }}
    if (msg.type==='options_applied') {{
      if (Array.isArray(msg.options)) renderOptions(msg.options);
      return;
    }}
    if (msg.type==='hello') {{
      setGPU(!!msg.cuda);
      return;
    }}
    if (msg.type==='start') {{
      W=msg.sheet_w; H=msg.sheet_h; M=msg.margin; setGroup(msg.group||'—'); setStatus('Starting…');
      sheets.clear(); sheetSel.innerHTML=''; currentSheet=0; setProgress(0,msg.total_parts||0); redraw();
      // show first sheet slot immediately
      setRunState(true);
      configMsg.textContent = 'Run in progress…';
      ensureSheet(0); setSheetPill(0, 1); redraw();
      return;
    }}
    if (msg.type==='group') {{
      setGroup(msg.group||'—'); setStatus('Group '+(msg.group||'—')); setProgress(0,msg.total_parts||0);
      sheets.clear(); sheetSel.innerHTML=''; currentSheet=0; ensureSheet(0); redraw();
      return;
    }}
    if (msg.type==='sheet_opened') {{
      ensureSheet(msg.sheet_index||0); setSheetPill(msg.sheet_index||0, msg.total_sheets||0);
      setStatus('Opened new sheet '+(1+(msg.sheet_index||0)));
      redraw();
      return;
    }}
    if (msg.type==='place') {{
      ensureSheet(msg.sheet);
      const rec = sheets.get(msg.sheet);
      if (msg.loops) for (const lp of msg.loops) rec.paths.push(lp);
      setProgress(msg.placed||0, msg.total||0);
      setStatus('Placed: '+(msg.part||'part')+'  (sheet '+(msg.sheet+1)+')');
      redraw();
      return;
    }}
    if (msg.type==='violations') {{
      if (msg.total>0) {{
        setStatus('Gap violations: '+msg.total+' (min gap failed on some pixels)');
      }} else {{
        setStatus('Gap check passed.');
      }}
      return;
    }}
    if (msg.type==='progress') {{
      if (msg.text) setStatus(msg.text);
      return;
    }}
    if (msg.type==='done' || msg.type==='stopped') {{
      setStatus(msg.type==='done' ? 'Completed.' : 'Stopped — partial saved.');
      setRunState(false);
      startBtn.disabled = true;
      configMsg.textContent = msg.type==='done' ? 'Completed.' : 'Stopped — partial saved.';
      if (msg.outputs) {{
        document.getElementById('outputs').textContent = (msg.outputs||[]).map(o=>o[0]+'  (sheets:'+o[1]+')').join('\\n') || '—';
      }}
      return;
    }}
  }} catch(e) {{ console.warn(e); }}
}};
</script>
</body></html>"""
    with open(path_on_disk, "w", encoding="utf-8") as f:
        f.write(html)

class NestHTTPHandler(BaseHTTPRequestHandler):
    hub:SSEHub = None
    control:NestControl = None
    folder:str = None
    ui_path:str = None
    cuda_on:bool = False

    def _set_headers(self, code=200, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        if extra:
            for k,v in (extra.items() if isinstance(extra,dict) else []): self.send_header(k,v)
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path)
        if p.path in ("/", "/index.html"):
            try:
                with open(self.ui_path, "r", encoding="utf-8") as f:
                    data = f.read().encode("utf-8")
                self._set_headers(200, "text/html; charset=utf-8"); self.wfile.write(data)
            except Exception as e:
                self._set_headers(500, "text/plain"); self.wfile.write(str(e).encode("utf-8"))
            return

        if p.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                # hello
                hello = json.dumps({"type":"hello","cuda":self.cuda_on})
                self.wfile.write(b"data: "); self.wfile.write(hello.encode("utf-8")); self.wfile.write(b"\n\n"); self.wfile.flush()
                # attach
                self.hub.attach(self)
                # keep open
                while True:
                    time.sleep(15)
                    try:
                        self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                    except Exception:
                        break
            finally:
                self.hub.detach(self)
            return

        if p.path == "/status":
            st = self.control.get_status()
            self._set_headers(200, "application/json"); self.wfile.write(json.dumps(st).encode("utf-8")); return

        if p.path == "/config":
            payload = {"options": _ui_toggle_snapshot(), "status": self.control.get_status()}
            self._set_headers(200, "application/json"); self.wfile.write(json.dumps(payload).encode("utf-8")); return

        self._set_headers(404, "text/plain"); self.wfile.write(b"Not found")

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/control":
            qs = parse_qs(p.query or "")
            cmd = (qs.get("cmd",[""])[0] or "").lower()
            if cmd == "pause":
                self.control.pause.set(); self._set_headers(200,"text/plain"); self.wfile.write(b"OK"); return
            if cmd == "resume":
                self.control.pause.clear(); self._set_headers(200,"text/plain"); self.wfile.write(b"OK"); return
            if cmd == "stop":
                self.control.stop.set();  self._set_headers(200,"text/plain"); self.wfile.write(b"OK"); return
            if cmd == "start":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except Exception:
                    length = 0
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    payload = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    self._set_headers(400, "text/plain"); self.wfile.write(b"Invalid JSON"); return
                if not isinstance(payload, dict):
                    self._set_headers(400, "text/plain"); self.wfile.write(b"JSON body must be an object"); return
                if not self.control.request_start(payload):
                    self._set_headers(409, "text/plain"); self.wfile.write(b"Already started"); return
                self._set_headers(200, "application/json"); self.wfile.write(json.dumps({"status":"starting"}).encode("utf-8"))
                self.hub.broadcast("starting", {"options": payload})
                return
            self._set_headers(400,"text/plain"); self.wfile.write(b"Bad command"); return
        self._set_headers(404,"text/plain"); self.wfile.write(b"Not found")

def start_http_server(folder:str, ui_filename:str, cuda_on:bool, control:NestControl, hub:SSEHub,
                      port:int=0, host:str="127.0.0.1"):
    ui_path = os.path.join(folder, ui_filename)
    write_standalone_html(ui_path)

    class _Server(ThreadingHTTPServer): daemon_threads=True
    NestHTTPHandler.hub = hub
    NestHTTPHandler.control = control
    NestHTTPHandler.folder = folder
    NestHTTPHandler.ui_path = ui_path
    NestHTTPHandler.cuda_on = cuda_on
    srv = _Server((host, port), NestHTTPHandler)
    host_bound, real_port = srv.server_address
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, host_bound, real_port

# ---------- placement (with live events + pause/stop checks) ----------
def bl_place(occ, mask_segments, tw):
    H=len(occ); W=len(occ[0]) if H>0 else 0
    ph=len(mask_segments)
    if H < ph or W < tw:
        return None
    max_y = H - ph + 1
    max_x = W - tw + 1
    for y in range(max_y):
        rows = occ[y:y+ph]
        x = 0
        while x < max_x:
            skip = 1
            blocked = False
            for yy, segs in enumerate(mask_segments):
                if not segs:
                    continue
                row = rows[yy]
                # find any occupied pixel in the segment window
                for start, end in segs:
                    # manual scan
                    for xx in range(x+start, x+end):
                        if row[xx]:
                            skip = max(skip, xx - (x + start) + 1)
                            blocked = True
                            break
                    if blocked: break
                if blocked: break
            if not blocked:
                return (x, y)
            x += skip
    return None

def or_mask_inplace(occ, mask_segments, mask_fills, ox, oy):
    for y,segs in enumerate(mask_segments):
        if not segs: continue
        row=occ[oy+y]
        fills=mask_fills[y]
        for (start,end),fill in zip(segs,fills):
            if not fill:
                continue
            dst_start=ox+start
            row[dst_start:dst_start + (end-start)] = fill

def pack_bitmap_core(ordered_parts: List['Part'], W: float, H: float, spacing: float, scale: int,
                     progress=None, progress_total=None, progress_prefix="",
                     mask_ops: Optional[Any] = None,
                     control: Optional[NestControl] = None,
                     event_sink: Optional[callable] = None):
    Wpx=max(1,int(math.ceil(W*scale))); Hpx=max(1,int(math.ceil(H*scale)))
    sheets_occ_raw=[]; sheets_occ_safe=[]; sheets_out=[]; sheets_count=0
    def ensure_sheet():
        nonlocal sheets_count
        if len(sheets_occ_raw)<=sheets_count:
            if mask_ops:
                sheets_occ_raw.append(mask_ops.zeros(Hpx,Wpx)); sheets_occ_safe.append(mask_ops.zeros(Hpx,Wpx))
            else:
                sheets_occ_raw.append(_empty_mask(Wpx,Hpx));     sheets_occ_safe.append(_empty_mask(Wpx,Hpx))
            sheets_out.append([])
        return sheets_occ_raw[sheets_count], sheets_occ_safe[sheets_count], sheets_out[sheets_count]

    placed_count=0; total_parts=progress_total if (progress_total is not None) else len(ordered_parts)

    def check_ctrl():
        if control:
            while control.pause.is_set(): time.sleep(0.05)
            if control.stop.is_set():
                partial=[{'sheet': i, 'loops': pl['loops']} for i,out in enumerate(sheets_out) for pl in out]
                used=max((pl['sheet'] for out in sheets_out for pl in out), default=-1)+1
                raise NestAbortPartial(partial, used)

    for p in ordered_parts:
        check_ctrl()
        placed=False
        for ang,mirror in p.candidate_poses():
            check_ctrl()
            cand=_get_part_candidate(p, scale, ang, mirror, spacing, ALLOW_NEST_IN_HOLES, ENFORCE_GAP)
            tensors=_ensure_mask_tensors(cand, mask_ops) if mask_ops else None
            attempt_sheet=sheets_count
            while True:
                check_ctrl()
                occ_raw,occ_safe,outlist=ensure_sheet()
                pos = (mask_ops.find_first_fit(occ_safe, tensors['test']) if mask_ops and tensors

                       else bl_place(occ_safe, cand['test_segments'], cand['pw']))

                if pos is not None:
                    xpx,ypx=pos
                    if mask_ops:
                        mask_ops.or_mask(occ_raw, tensors['raw'], xpx, ypx)
                        mask_ops.or_mask(occ_safe, tensors['occ'], xpx, ypx)
                    else:
                        or_mask_inplace(occ_raw, cand['raw_segments'], cand['raw_fills'], xpx, ypx)
                        or_mask_inplace(occ_safe,cand['occ_segments'], cand['occ_fills'], xpx, ypx)

                    x_units=xpx/scale; y_units=ypx/scale

                    loops_t=[[ (x+x_units,y+y_units) for (x,y) in lp ] for lp in cand['loops']]
                    outlist.append({'sheet':sheets_count,'loops':loops_t})
                    placed=True; placed_count+=1
                    if event_sink:
                        event_sink("place", {"sheet":sheets_count,"loops":loops_t,"part":os.path.basename(p.name),
                                             "placed":placed_count,"total":total_parts})
                    if progress:
                        progress(f"{progress_prefix}Placing parts…\nPlaced: {placed_count}/{total_parts}\nCurrent sheet: {sheets_count+1}\nPart: {os.path.basename(p.name)}")
                    break
                else:
                    sheets_count+=1
                    if event_sink:
                        event_sink("sheet_opened", {"sheet_index": sheets_count})
                    if progress:
                        progress(f"{progress_prefix}Opening new sheet… now {sheets_count+1}\nPlaced: {placed_count}/{total_parts}")
                    if sheets_count>attempt_sheet+25: break
            if placed: break

        if not placed:
            sheets_count+=1
            occ_raw,occ_safe,outlist=ensure_sheet()
            ang,mirror=0.0,False
            cand=_get_part_candidate(p, scale, ang, mirror, spacing, ALLOW_NEST_IN_HOLES, ENFORCE_GAP)
            tensors=_ensure_mask_tensors(cand, mask_ops) if mask_ops else None
            if mask_ops and tensors:
                mask_ops.or_mask(occ_raw,tensors['raw'],0,0)
                mask_ops.or_mask(occ_safe,tensors['occ'],0,0)
            else:
                or_mask_inplace(occ_raw,cand['raw_segments'],cand['raw_fills'],0,0)
                or_mask_inplace(occ_safe,cand['occ_segments'],cand['occ_fills'],0,0)

            loops_t=[[ (x,y) for (x,y) in lp ] for lp in cand['loops']]

            outlist.append({'sheet':sheets_count,'loops':loops_t})
            placed_count+=1
            if event_sink:
                event_sink("place", {"sheet":sheets_count,"loops":loops_t,
                                     "part":os.path.basename(p.name),"placed":placed_count,"total":total_parts})
            if progress:
                progress(f"Forced place on new sheet {sheets_count+1}\nPlaced: {placed_count}/{total_parts}")

    used_sheets = max((pl['sheet'] for out in sheets_out for pl in out), default=-1)+1
    if mask_ops:
        fill_pixels = 0
        for occ in sheets_occ_raw: fill_pixels += mask_ops.count_true(occ)
    else:
        fill_pixels = sum(sum(1 for v in row if v) for occ in sheets_occ_raw for row in occ)

    placements=[{'sheet':i,'loops':pl['loops']} for i,out in enumerate(sheets_out) for pl in out]
    return placements, used_sheets, fill_pixels

def _seq_key(order: List['Part']): return tuple(p.uid for p in order)

def _result_is_better(candidate, incumbent):
    if candidate is None: return False
    if incumbent is None: return True
    _, cs, cf = candidate; _, is_, if_ = incumbent
    if cs != is_: return cs < is_
    return cf > if_

def _mutate_order(order: List['Part'], rnd: Random) -> List['Part']:
    n=len(order)
    if n<=1: return list(order)
    op=rnd.random()
    if n==2: op=0.0
    if op<0.4:
        i,j=rnd.sample(range(n),2); new=list(order); new[i],new[j]=new[j],new[i]; return new
    elif op<0.75:
        i,j=rnd.sample(range(n),2); new=list(order); part=new.pop(i); new.insert(j,part); return new
    else:
        i,j=sorted(rnd.sample(range(n),2)); new=list(order); new[i:j+1]=reversed(new[i:j+1]); return new

def _anneal_order(initial_order: List['Part'], evaluate_fn, rnd: Random, sheet_penalty: int,
                  progress=None, label="", max_iters: Optional[int] = None, control: Optional[NestControl]=None):
    order=list(initial_order); best_order=list(order); best_result=evaluate_fn(best_order, allow_progress=False)
    current_order=list(order); current_result=best_result
    n=len(order)
    if n<=1: return best_order, best_result
    default_iters=max(8,min(24,n+4)); base_iters=max(5,min(default_iters,max_iters)) if max_iters is not None else default_iters
    temperature=max(1.0,n*0.4); cooling=0.9; stall_limit=None; stall=0
    def score(res):
        if res is None: return float('inf')
        _,sh,fi=res; return sh*sheet_penalty - fi
    for it in range(1, base_iters+1):
        if control and control.stop.is_set(): break
        while control and control.pause.is_set(): time.sleep(0.05)
        cand_order=_mutate_order(current_order,rnd)
        cand_result=evaluate_fn(cand_order, allow_progress=False)
        if _result_is_better(cand_result,current_result):
            current_order,current_result=cand_order,cand_result
        else:
            delta=score(cand_result)-score(current_result)
            accept_prob=1.0 if delta<0 else (math.exp(-delta/temperature) if temperature>0 else 0.0)
            if accept_prob>rnd.random(): current_order,current_result=cand_order,cand_result
        if _result_is_better(current_result,best_result):
            best_order,best_result=list(current_order),current_result
            if progress: progress(f"{label}Anneal improvement: sheets={best_result[1]}, fill={best_result[2]}")
            stall=0
        else: stall+=1
        if progress and it % max(6, base_iters//3)==0:
            progress(f"{label}Anneal {it}/{base_iters}: best sheets={best_result[1]}, fill={best_result[2]}")
        temperature*=cooling
        if temperature<1e-4: temperature=1e-4
        if stall_limit is None: stall_limit=max(3, base_iters//2)
        if stall>=stall_limit: break
    return best_order, best_result

def pack_bitmap_multi(parts: List['Part'], W: float, H: float, spacing: float, scale: int,
                      tries: int, seed: Optional[int], progress=None,
                      mask_ops: Optional[Any] = None,
                      control: Optional[NestControl]=None,
                      event_sink: Optional[callable]=None):
    base=[p for p in parts if p.outer is not None]
    base.sort(key=lambda p: abs(polygon_area(p.outer)), reverse=True)
    rnd=Random(seed) if seed is not None else Random()
    total_parts=len(base)
    if total_parts==0: return [], 0
    search_scale=scale
    if scale > 6:
        search_scale = max(6, scale // 2)  # coarse for trials → faster

    Wpx=max(1,int(math.ceil(W*scale))); Hpx=max(1,int(math.ceil(H*scale))); sheet_penalty=Wpx*Hpx*1000
    cache: Dict[Tuple[tuple,int], Tuple[List[dict],int,int]] = {}

    def evaluate(order: List['Part'], allow_progress: bool, prefix: str = "", use_scale: int = search_scale):
        key=(_seq_key(order),use_scale)
        if key in cache: return cache[key]
        res=pack_bitmap_core(order,W,H,spacing,use_scale,
                             progress=(progress if allow_progress else None),
                             progress_total=total_parts if allow_progress else None,
                             progress_prefix=prefix if allow_progress else "",
                             mask_ops=mask_ops, control=control, event_sink=event_sink)
        cache[key]=res; return res

    best_result=None; best_order=None
    heuristic=[("Area-desc ", list(base)),
               ("Aspect-desc ", sorted(base, key=lambda p: max(p.w,p.h,p.obb_w,p.obb_h), reverse=True)),
               ("Tall-first ",  sorted(base, key=lambda p: p.h, reverse=True))]
    tries=max(1,tries)
    starts=[]
    for ho in heuristic:
        if len(starts)>=tries: break
        starts.append(ho)
    while len(starts)<tries:
        idx=len(starts)-len(heuristic)+1
        starts.append((f"Random {max(1,idx)} ", rnd.sample(base, len(base))))

    attempts=max(1,len(starts))
    anneal_limit=max(4,min(8,total_parts+max(1,tries//2)))
    last_start=None
    for t,(label,start_order) in enumerate(starts):
        if progress: progress(f"{label}placement trial {t+1}/{attempts}…")
        last_start=evaluate(start_order, allow_progress=False, prefix=f"{label}Try {t+1}/{attempts}\n", use_scale=search_scale)
        if anneal_limit<=0:
            order_after,result_after=start_order,last_start
        else:
            limit=anneal_limit if t==0 else (min(3,anneal_limit) if t<len(heuristic) else min(4,anneal_limit))
            if limit<=1:
                order_after,result_after=start_order,last_start
            else:
                order_after,result_after=_anneal_order(start_order,
                    lambda o, allow_progress=False: evaluate(o, allow_progress, prefix=label, use_scale=search_scale),
                    rnd, sheet_penalty, progress=progress, label=label, max_iters=limit, control=control)
        final = result_after if _result_is_better(result_after,last_start) else last_start
        final_order = order_after if final is result_after else start_order
        if _result_is_better(final,best_result):
            best_result=final; best_order=final_order
            if progress: progress(f"{label}New global best: sheets={best_result[1]}, fill={best_result[2]}")
        elif progress and best_result:
            progress(f"{label}Result sheets={final[1]}, fill={final[2]} (best remains sheets={best_result[1]}, fill={best_result[2]})")
    if best_result is None:
        best_result=last_start; best_order=starts[0][1] if starts else base
    final_order=best_order if best_order is not None else base
    final_result=evaluate(final_order, allow_progress=True, prefix="Final pass\n", use_scale=scale)
    return final_result[0], final_result[1]

# ---------- Shelf fallback ----------
def pack_shelves(parts: List['Part'], W: float, H: float, spacing: float,
                 control: Optional[NestControl]=None, event_sink: Optional[callable]=None,
                 scale: Optional[int] = None):
    parts=sorted([p for p in parts if p.outer is not None], key=lambda p: max(p.w,p.h,p.obb_w,p.obb_h), reverse=True)
    if not parts:
        return [], 0

    scale_eff = max(1, int(scale) if scale else _eff_scale(PIXELS_PER_UNIT, spacing))
    Wpx=max(1,int(math.ceil(W*scale_eff)))
    Hpx=max(1,int(math.ceil(H*scale_eff)))
    spacing_px=int(math.ceil(spacing*scale_eff)) if spacing>0 else 0

    placements=[]; sheet=0; shelf_y_px=0; shelf_h_px=0; cursor_x_px=0
    occ_masks: List[List[bytearray]] = []

    def ensure_occ(idx: int):
        while len(occ_masks) <= idx:
            occ_masks.append(_empty_mask(Wpx, Hpx))
        return occ_masks[idx]

    def new_sheet():
        nonlocal sheet, shelf_y_px, shelf_h_px, cursor_x_px
        sheet += 1
        shelf_y_px = 0
        shelf_h_px = 0
        cursor_x_px = 0
        ensure_occ(sheet)
        if event_sink:
            event_sink("sheet_opened", {"sheet_index": sheet})

    def fits(sheet_idx: int, cand: Dict[str, Any], ox: int, oy: int) -> bool:
        occ = ensure_occ(sheet_idx)
        test_segments = cand['test_segments'] if ENFORCE_GAP else cand['occ_segments']
        Hcur = len(occ)
        Wcur = len(occ[0]) if Hcur else 0
        for yy, segs in enumerate(test_segments):
            if not segs:
                continue
            dst_y = oy + yy
            if dst_y < 0 or dst_y >= Hcur:
                return False
            row = occ[dst_y]
            for start, end in segs:
                dst_start = ox + start
                dst_end = ox + end
                if dst_start < 0 or dst_end > Wcur:
                    return False
                for xx in range(dst_start, dst_end):
                    if row[xx]:
                        return False
        return True

    def commit(sheet_idx: int, cand: Dict[str, Any], ox: int, oy: int):
        occ = ensure_occ(sheet_idx)
        or_mask_inplace(occ, cand['occ_segments'], cand['occ_fills'], ox, oy)

    ensure_occ(sheet)
    total=len(parts)

    for idx,p in enumerate(parts,1):
        if control and control.stop.is_set():
            raise NestAbortPartial(placements, (max((pl['sheet'] for pl in placements), default=-1))+1 )
        while control and control.pause.is_set():
            time.sleep(0.05)

        cand_opts=[]
        for ang, mirror in p.candidate_poses():
            cand=_get_part_candidate(p, scale_eff, ang, mirror, spacing, ALLOW_NEST_IN_HOLES, ENFORCE_GAP)
            cand_opts.append((ang, mirror, cand))

        placed=False
        for ang, mirror, cand in cand_opts:

            if (cursor_x_px + cand['pw'] + spacing_px <= Wpx and
                shelf_y_px + max(shelf_h_px, cand['ph'] + spacing_px) <= Hpx and
                fits(sheet, cand, cursor_x_px, shelf_y_px)):
                x_units=cursor_x_px/scale_eff; y_units=shelf_y_px/scale_eff

                loops_t=[[ (x+x_units,y+y_units) for x,y in lp ] for lp in cand['loops']]
                placements.append({'sheet':sheet,'loops':loops_t})
                if event_sink:
                    event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),
                                          "placed":idx,"total":total})
                commit(sheet, cand, cursor_x_px, shelf_y_px)

                cursor_x_px += cand['pw'] + spacing_px
                shelf_h_px = max(shelf_h_px, cand['ph'] + spacing_px)

                placed=True
                break
        if placed:
            continue

        shelf_y_px += shelf_h_px
        cursor_x_px = 0
        shelf_h_px = 0
        for ang, mirror, cand in cand_opts:

            if (cand['pw'] + spacing_px <= Wpx and
                shelf_y_px + cand['ph'] + spacing_px <= Hpx and
                fits(sheet, cand, cursor_x_px, shelf_y_px)):
                x_units=cursor_x_px/scale_eff; y_units=shelf_y_px/scale_eff

                loops_t=[[ (x+x_units,y+y_units) for x,y in lp ] for lp in cand['loops']]
                placements.append({'sheet':sheet,'loops':loops_t})
                if event_sink:
                    event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),
                                          "placed":idx,"total":total})
                commit(sheet, cand, cursor_x_px, shelf_y_px)

                cursor_x_px = cand['pw'] + spacing_px
                shelf_h_px = cand['ph'] + spacing_px

                placed=True
                break
        if placed:
            continue

        new_sheet()
        ok=False
        for ang, mirror, cand in cand_opts:

            if (cand['pw'] + spacing_px <= Wpx and
                cand['ph'] + spacing_px <= Hpx and
                fits(sheet, cand, 0, 0)):
                x_units=0.0; y_units=0.0

                loops_t=[[ (x+x_units,y+y_units) for x,y in lp ] for lp in cand['loops']]
                placements.append({'sheet':sheet,'loops':loops_t})
                if event_sink:
                    event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),
                                          "placed":idx,"total":total})
                commit(sheet, cand, 0, 0)

                cursor_x_px = cand['pw'] + spacing_px
                shelf_h_px = cand['ph'] + spacing_px

                ok=True
                break
        if not ok:
            cand=_get_part_candidate(p, scale_eff, 0.0, False, spacing, ALLOW_NEST_IN_HOLES, ENFORCE_GAP)
            commit(sheet, cand, 0, 0)

            loops_t=[[ (x,y) for x,y in lp ] for lp in cand['loops']]

            placements.append({'sheet':sheet,'loops':loops_t})
            if event_sink:
                event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),
                                      "placed":idx,"total":total})

            cursor_x_px = cand['pw'] + spacing_px
            shelf_h_px = cand['ph'] + spacing_px


    sheets_used=(max((pl['sheet'] for pl in placements), default=-1))+1
    return placements, sheets_used

# ---------- Gap validator ----------
def check_min_gap_violations(placements: List[dict], sheets_used: int, W: float, H: float, spacing: float, scale: int):
    """Returns total count of pixels that violate the spacing (approx), per-sheet counts, and first few sample coords."""
    if spacing <= 0: return 0, [0]*max(1,sheets_used), []
    Wpx=max(1,int(math.ceil(W*scale))); Hpx=max(1,int(math.ceil(H*scale)))
    r_val=max(0,int(math.ceil(spacing*scale)))
    per_sheet=[0 for _ in range(max(1,sheets_used))]
    samples=[]
    for s in range(max(1,sheets_used)):
        occ=_empty_mask(Wpx,Hpx)
        for pl in (p for p in placements if p['sheet']==s):
            raw,pw,ph,minx,miny = rasterize_loops_with_bbox(pl['loops'], scale)
            if r_val>0:
                raw = dilate_mask(raw,pw,ph,r_val)
            ox=int(math.floor(minx*scale)); oy=int(math.floor(miny*scale))
            for yy in range(ph):
                dst_y = oy+yy
                if dst_y<0 or dst_y>=Hpx: continue
                src_row = raw[yy]
                dst_row = occ[dst_y]
                for xx in range(pw):
                    if src_row[xx]:
                        dst_x = ox+xx
                        if 0<=dst_x<Wpx:
                            if dst_row[dst_x]:
                                per_sheet[s]+=1
                                if len(samples)<16:
                                    samples.append((s, dst_x, dst_y))
                            else:
                                dst_row[dst_x]=1
    total=sum(per_sheet)
    return total, per_sheet, samples

# ---------- Line merging & writers ----------
def merge_common_lines(placements: List[dict], tol=1e-4) -> List[dict]:
    def norm_seg(a,b):
        ax,ay=a; bx,by=b
        if (bx<ax) or (abs(bx-ax)<=tol and by<ay): ax,ay,bx,by=bx,by,ax,ay
        return (round(ax/tol), round(ay/tol), round(bx/tol), round(by/tol))
    keep={}
    for pl in placements:
        new_loops=[]
        for lp in pl['loops']:
            segs=[]
            for i in range(len(lp)-1):
                a=lp[i]; b=lp[i+1]
                key=norm_seg(a,b); keep[key]=keep.get(key,0)+1; segs.append((a,b))
            new_loops.append(segs)
        pl['__segs__']=new_loops
    out=[]
    for pl in placements:
        lines=[]
        for segs in pl['__segs__']:
            for a,b in segs:
                key=norm_seg(a,b)
                if keep.get(key,0)==1: lines.append((a,b))
        out.append({'sheet': pl['sheet'], 'lines': lines})
    return out

def write_r12_dxf(path, sheets, W, H, placements, margin, merge_lines=False):
    def w(f,c,v): f.write(f"{c}\n{v}\n")
    with open(path,'w',encoding='utf-8') as f:
        w(f,0,"SECTION"); w(f,2,"HEADER"); w(f,9,"$INSUNITS"); w(f,70,INSUNITS); w(f,0,"ENDSEC")
        w(f,0,"SECTION"); w(f,2,"TABLES"); w(f,0,"ENDSEC")
        w(f,0,"SECTION"); w(f,2,"ENTITIES")
        for s in range(sheets):
            sheet_ox = s*(W + 2*margin + SHEET_GAP)
            w(f,0,"POLYLINE"); w(f,8,"SHEET"); w(f,66,1); w(f,70,1)
            for x,y in [(sheet_ox,0),(sheet_ox+W+2*margin,0),(sheet_ox+W+2*margin,H+2*margin),(sheet_ox,H+2*margin),(sheet_ox,0)]:
                w(f,0,"VERTEX"); w(f,8,"SHEET"); w(f,10,x); w(f,20,y)
            w(f,0,"SEQEND")
        if merge_lines:
            merged = merge_common_lines(placements)
            for pl in merged:
                ox = pl['sheet']*(W + 2*margin + SHEET_GAP) + margin
                oy = margin
                for (a,b) in pl['lines']:
                    w(f,0,"LINE"); w(f,8,"NEST")
                    w(f,10,a[0]+ox); w(f,20,a[1]+oy)
                    w(f,11,b[0]+ox); w(f,21,b[1]+oy)
        else:
            for pl in placements:
                ox = pl['sheet']*(W + 2*margin + SHEET_GAP) + margin
                oy = margin
                for lp in pl['loops']:
                    w(f,0,"POLYLINE"); w(f,8,"NEST"); w(f,66,1); w(f,70,1)
                    for x,y in ((x+ox,y+oy) for x,y in lp):
                        w(f,0,"VERTEX"); w(f,8,"NEST"); w(f,10,x); w(f,20,y)
                    w(f,0,"SEQEND")
        w(f,0,"ENDSEC"); w(f,0,"EOF")

def write_split_sheets(base_path_no_ext: str, placements: List[dict], total_sheets: int,
                       W: float, H: float, margin: float, merge_lines=False):
    for s in range(total_sheets):
        sub=[{'sheet':0,'loops':pl['loops']} for pl in placements if pl['sheet']==s]
        out=f"{base_path_no_ext}-s{s+1}.dxf"
        write_r12_dxf(out,1,W,H,sub,margin,merge_lines=merge_lines)

# ---------- main ----------
def main_live():
    prog = WinProgress("Nesting DXF… (HTML live viewer)", 520, 220); prog.create()

    if not os.path.isdir(FOLDER):
        log(f"[ERROR] Folder not found: {FOLDER}"); prog.update("Folder not found."); prog.close(); return

    dxf_files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(".dxf") and f.lower()!="nested.dxf"])
    if not dxf_files:
        log(f"[WARN] No .dxf files found in: {FOLDER}"); prog.update("No .dxf files found."); prog.close(); return

    W_eff=SHEET_W-2*SHEET_MARGIN; H_eff=SHEET_H-2*SHEET_MARGIN
    if W_eff<=0 or H_eff<=0:
        msg=f"[ERROR] SHEET_MARGIN={SHEET_MARGIN} leaves no usable area on a {SHEET_W}×{SHEET_H} sheet."
        log(msg); prog.update(msg); prog.close(); return

    eff_scale = _eff_scale(PIXELS_PER_UNIT, SPACING)

    # Prepare control + SSE
    control = NestControl()
    hub = SSEHub()

    # accelerator
    mask_ops = build_mask_ops(BITMAP_DEVICE)
    accel_note = "Acceleration: CPU bitmap evaluator"; using_cuda=False
    if mask_ops:
        dev = getattr(mask_ops, "device", "cpu"); dev_type = getattr(dev, "type", str(dev))
        if str(dev_type).lower()=="cuda": using_cuda=True; accel_note=f"Acceleration: CUDA GPU ({dev}) via PyTorch"
        elif str(dev).lower()=="numpy": accel_note = "Acceleration: NumPy (CPU)"
        else: accel_note=f"Acceleration: PyTorch device {dev}"

    # start HTTP server + open viewer
    srv, bound_host, port = start_http_server(FOLDER, UI_FILENAME, using_cuda, control, hub, HTTP_PORT, HTTP_HOST)
    open_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::", "") else bound_host
    url=f"http://{open_host}:{port}/"
    try: webbrowser.open(url)
    except: pass
    if bound_host != open_host:
        log(f"[INFO] Live viewer at: {url} (server bound to {bound_host})")
    else:
        log(f"[INFO] Live viewer at: {url}")

    wait_text = "Viewer ready — adjust options in the browser and press Start."
    control.set_status(phase="waiting")
    hub.broadcast("waiting", {"message": wait_text, "options": _ui_toggle_snapshot()})
    prog.update(wait_text)

    # status helpers
    def progress_cb(text: str):
        prog.update(text)
        hub.broadcast("progress", {"text": text})

    def event_sink(kind: str, payload: dict):
        if kind=="place":
            hub.broadcast("place", payload)
        elif kind=="sheet_opened":
            hub.broadcast("sheet_opened", payload)

    # Wait for the UI to kick off the run
    start_config = control.wait_for_start()
    _apply_toggle_config(start_config)
    applied_opts = _ui_toggle_snapshot()
    hub.broadcast("options_applied", {"options": applied_opts})
    for opt in applied_opts:
        log(f"[INFO] {opt['label']}: {'ON' if opt['value'] else 'OFF'}")

    control.set_status(phase="reading")
    hub.broadcast("progress", {"text":"Reading DXFs…"})
    prog.update(f"Reading DXFs… 0/{len(dxf_files)}")

    # parse all parts + groups
    all_parts=[]; groups={}; skipped=0
    for idx,fn in enumerate(dxf_files,1):
        prog.update(f"Reading DXFs… {idx}/{len(dxf_files)}  {fn}")
        path=os.path.join(FOLDER,fn)
        loops,segs=parse_entities(path)
        if not loops and segs: loops=join_segments_to_loops(segs,JOIN_TOL)
        fallback_bbox=None
        if not loops and segs and FALLBACK_OPEN_AS_BBOX:
            pts=[pt for a,b in segs for pt in (a,b)]
            if pts:
                minx,miny,maxx,maxy=bbox_of_points(pts)
                if maxx>minx and maxy>miny: fallback_bbox=(minx,miny,maxx,maxy)
        if not loops and fallback_bbox is None:
            log(f"[WARN] {fn}: no closed loops; skipped."); skipped+=1; continue
        p=Part(fn,loops,fallback_bbox)
        if p.outer is None or p.w<=0 or p.h<=0:
            log(f"[WARN] {fn}: zero-sized; skipped."); skipped+=1; continue
        qty=read_qty_for_dxf(FOLDER,fn)
        thk_label=read_thickness_label(FOLDER,fn,THICKNESS_LABEL_UNITS)
        for _ in range(qty):
            all_parts.append(p); groups.setdefault(thk_label,[]).append(p)

    if not all_parts:
        log("[WARN] Nothing to nest.")
        hub.broadcast("progress", {"text":"Nothing to nest."})
        control.set_status(phase="done")
        hub.broadcast("done", {"outputs": []})
        prog.update("Nothing to nest."); prog.close(); return

    outputs=[]
    hub.broadcast("hello", {"cuda": using_cuda})

    def do_one_batch(parts: List[Part], group_label: str):
        total=len(parts)
        hub.broadcast("start", {"sheet_w":W_eff,"sheet_h":H_eff,"margin":SHEET_MARGIN,"total_parts":total,"group":group_label})
        hub.broadcast("sheet_opened", {"sheet_index": 0})
        control.set_status(phase="nest", group=group_label, total=total, placed=0)
        if NEST_MODE.lower()=="bitmap":
            try:
                if SHUFFLE_TRIES>1:
                    placements, sheets = pack_bitmap_multi(parts, W_eff, H_eff, SPACING, eff_scale,
                                                           SHUFFLE_TRIES, SHUFFLE_SEED,
                                                           progress=progress_cb, mask_ops=mask_ops,
                                                           control=control, event_sink=event_sink)
                else:
                    res = pack_bitmap_core(parts, W_eff, H_eff, SPACING, eff_scale,
                                           progress=progress_cb, progress_total=len(parts),
                                           mask_ops=mask_ops, control=control, event_sink=event_sink)
                    placements, sheets = res[0], res[1]
            except NestAbortPartial as nb:
                placements, sheets = nb.placements, nb.sheets
                # validate even on stop, if requested
                if ENFORCE_GAP:
                    total_v, per_sheet_v, _ = check_min_gap_violations(placements, sheets, W_eff, H_eff, SPACING, eff_scale)
                    hub.broadcast("violations", {"total": int(total_v), "per_sheet": per_sheet_v})
                    log(f"[CHECK] Gap violations (stopped): total={total_v} per_sheet={per_sheet_v}")
                if SPLIT_SHEETS:
                    base_no_ext=os.path.join(FOLDER, f"{group_label}-nested")
                    write_split_sheets(base_no_ext, placements, sheets, W_eff, H_eff, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((base_no_ext+"-s*", sheets))
                else:
                    out=os.path.join(FOLDER, f"{group_label}-nested.dxf")
                    write_r12_dxf(out, sheets, W_eff, H_eff, placements, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((out, sheets))
                hub.broadcast("stopped", {"outputs": outputs})
                control.set_status(phase="stopped")
                return False  # stopped
        else:
            try:
                placements, sheets = pack_shelves(parts, W_eff, H_eff, SPACING,
                                                control=control, event_sink=event_sink,
                                                scale=eff_scale)
            except NestAbortPartial as nb:
                placements, sheets = nb.placements, nb.sheets
                if ENFORCE_GAP:
                    total_v, per_sheet_v, _ = check_min_gap_violations(placements, sheets, W_eff, H_eff, SPACING, eff_scale)
                    hub.broadcast("violations", {"total": int(total_v), "per_sheet": per_sheet_v})
                    log(f"[CHECK] Gap violations (stopped): total={total_v} per_sheet={per_sheet_v}")
                if SPLIT_SHEETS:
                    base_no_ext=os.path.join(FOLDER, f"{group_label}-nested")
                    write_split_sheets(base_no_ext, placements, sheets, W_eff, H_eff, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((base_no_ext+"-s*", sheets))
                else:
                    out=os.path.join(FOLDER, f"{group_label}-nested.dxf")
                    write_r12_dxf(out, sheets, W_eff, H_eff, placements, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((out, sheets))
                hub.broadcast("stopped", {"outputs": outputs})
                control.set_status(phase="stopped")
                return False

        if sheets<=0:
            log(f"[WARN] Parts @ {group_label} exist, but none fit.")
            return True

        # Post-run validator (approximate raster validator)
        if ENFORCE_GAP:
            total_v, per_sheet_v, _ = check_min_gap_violations(placements, sheets, W_eff, H_eff, SPACING, eff_scale)
            hub.broadcast("violations", {"total": int(total_v), "per_sheet": per_sheet_v})
            if total_v>0:
                log(f"[CHECK] Gap validator: violations found — total={total_v}; per-sheet={per_sheet_v}")
            else:
                log("[CHECK] Gap validator: PASS")

        if SPLIT_SHEETS:
            base_no_ext=os.path.join(FOLDER, f"{group_label}-nested")
            write_split_sheets(base_no_ext, placements, sheets, W_eff, H_eff, SHEET_MARGIN, merge_lines=MERGE_LINES)
            outputs.append((base_no_ext+"-s*", sheets))
        else:
            out=os.path.join(FOLDER, f"{group_label}-nested.dxf")
            write_r12_dxf(out, sheets, W_eff, H_eff, placements, SHEET_MARGIN, merge_lines=MERGE_LINES)
            outputs.append((out, sheets))
        return True

    if GROUP_BY_THICKNESS:
        for thk_label, parts in sorted(groups.items(), key=lambda kv: kv[0]):
            hub.broadcast("group", {"group": thk_label, "total_parts": len(parts)})
            ok = do_one_batch(parts, thk_label)
            if not ok: break
    else:
        ok = do_one_batch(all_parts, "all")

    # Report
    report_path = os.path.join(FOLDER, "nest_report.txt")
    _report_lines.insert(0,"=== Nesting Report ===")
    for out,sheets in outputs: _report_lines.append(f"Saved: {out}  | Sheets: {sheets}")
    _report_lines.append(f"Mode: {NEST_MODE}")
    _report_lines.append(f"Margin: {SHEET_MARGIN}")
    _report_lines.append(f"Spacing: {SPACING}")
    _report_lines.append(f"Resolution (eff): {eff_scale} px/unit")
    _report_lines.append(f"Shuffle tries: {SHUFFLE_TRIES}{'' if SHUFFLE_SEED is None else f' (seed {SHUFFLE_SEED})'}")
    _report_lines.append(f"Rect-align mode: {RECT_ALIGN_MODE}")
    _report_lines.append(f"Allow mirror: {ALLOW_MIRROR}")
    _report_lines.append(f"Allow nest in holes: {ALLOW_NEST_IN_HOLES}")
    _report_lines.append(f"Enforce gap: {ENFORCE_GAP}")
    _report_lines.append(f"Thickness label units: {THICKNESS_LABEL_UNITS}")
    _report_lines.append(f"Split sheets: {SPLIT_SHEETS}")
    _report_lines.append(f"Merge common lines: {MERGE_LINES}")
    _report_lines.append(f"INSUNITS header: {'inches' if INSUNITS==1 else 'mm'}")
    _report_lines.append(accel_note)
    try:
        with open(report_path,"w",encoding="utf-8") as rf: rf.write("\n".join(_report_lines))
    except Exception as e:
        print(f"[WARN] Could not write report: {e}")

    control.set_status(phase="done")
    hub.broadcast("done", {"outputs": outputs})
    prog.update("Done. You can close this window."); prog.close()

# ---------- crash log ----------
def _write_crash(folder: str, ex: BaseException):
    try:
        path=os.path.join(folder if os.path.isdir(folder) else os.getcwd(), "nest_error.txt")
        with open(path,"a",encoding="utf-8") as f:
            f.write(f"=== Crash {datetime.datetime.now().isoformat()} ===\n")
            traceback.print_exc(file=f); f.write("\n")
        print(f"[ERROR] A crash log was written to: {path}")
    except: pass

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DXF Nesting with Live HTML viewer (SSE).")
    parser.add_argument("--folder", default=FOLDER, help="Folder with DXFs and optional qty TXTs.")
    parser.add_argument("--sheet", nargs=2, metavar=("WIDTH","HEIGHT"), type=float, default=(SHEET_W, SHEET_H))
    parser.add_argument("--margin", type=float, default=SHEET_MARGIN)
    parser.add_argument("--spacing", type=float, default=SPACING)
    parser.add_argument("--nest-mode", choices=["bitmap","shelf"], default=NEST_MODE)
    parser.add_argument("--pixels-per-unit", type=int, default=PIXELS_PER_UNIT)
    parser.add_argument("--tries", type=int, default=SHUFFLE_TRIES)
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--workers", type=int, default=BITMAP_EVAL_WORKERS)
    parser.add_argument("--device", default=BITMAP_DEVICE, help="PyTorch device (e.g., 'cuda','cuda:0','cpu').")
    parser.add_argument("--allow-mirror", dest="allow_mirror", action="store_true", default=ALLOW_MIRROR)
    parser.add_argument("--no-mirror", dest="allow_mirror", action="store_false")
    parser.add_argument("--allow-hole-nesting", dest="allow_holes", action="store_true", default=ALLOW_NEST_IN_HOLES)
    parser.add_argument("--forbid-hole-nesting", dest="allow_holes", action="store_false")
    parser.add_argument("--rect-align", choices=["off","prefer","force"], default=RECT_ALIGN_MODE)
    parser.add_argument("--group-by-thickness", dest="group_by_thickness", action="store_true", default=GROUP_BY_THICKNESS)
    parser.add_argument("--no-group-by-thickness", dest="group_by_thickness", action="store_false")
    parser.add_argument("--thickness-label-units", choices=["auto","in","mm"], default=THICKNESS_LABEL_UNITS)
    parser.add_argument("--split-sheets", dest="split_sheets", action="store_true", default=SPLIT_SHEETS)
    parser.add_argument("--no-split-sheets", dest="split_sheets", action="store_false")
    parser.add_argument("--merge-lines", dest="merge_lines", action="store_true", default=MERGE_LINES)
    parser.add_argument("--no-merge-lines", dest="merge_lines", action="store_false")
    parser.add_argument("--rotation-step-deg", type=float, default=ROTATION_STEP_DEG)
    parser.add_argument("--insunits", choices=["in","mm"], default=("in" if INSUNITS==1 else "mm"))
    parser.add_argument("--port", type=int, default=HTTP_PORT, help="HTTP port for the viewer (0=auto).")
    parser.add_argument("--host", default=HTTP_HOST, help="Host/interface for the viewer server (default 127.0.0.1).")
    parser.add_argument("--pause-on-exit", dest="pause_on_exit", action="store_true", default=False)
    parser.add_argument("--enforce-gap", dest="enforce_gap", action="store_true", default=ENFORCE_GAP)
    parser.add_argument("--no-enforce-gap", dest="enforce_gap", action="store_false")
    args = parser.parse_args()

    # apply args
    FOLDER = os.path.abspath(args.folder)
    SHEET_W, SHEET_H = map(float, args.sheet)
    SHEET_MARGIN = float(args.margin); SPACING=float(args.spacing)
    NEST_MODE = args.nest_mode
    PIXELS_PER_UNIT = max(1, int(args.pixels_per_unit))
    SHUFFLE_TRIES = max(1, int(args.tries)); SHUFFLE_SEED=args.seed
    BITMAP_EVAL_WORKERS=args.workers; BITMAP_DEVICE=args.device
    ALLOW_MIRROR=args.allow_mirror; ALLOW_NEST_IN_HOLES=args.allow_holes
    RECT_ALIGN_MODE=args.rect_align; GROUP_BY_THICKNESS=args.group_by_thickness
    THICKNESS_LABEL_UNITS=args.thickness_label_units
    SPLIT_SHEETS=args.split_sheets; MERGE_LINES=args.merge_lines
    ROTATION_STEP_DEG=max(0.0, float(args.rotation_step_deg))
    INSUNITS = 1 if args.insunits=="in" else 4
    HTTP_PORT = int(args.port)
    HTTP_HOST = args.host
    ENFORCE_GAP = bool(args.enforce_gap)

    try:
        main_live()
    except Exception as ex:
        _write_crash(FOLDER, ex)
        traceback.print_exc()
    finally:
        if args.pause_on_exit and IS_WINDOWS:
            try: input("\nPress Enter to exit…")
            except: pass
