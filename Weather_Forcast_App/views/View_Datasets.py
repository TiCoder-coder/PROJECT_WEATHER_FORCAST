import mimetypes
from pathlib import Path
from datetime import datetime

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render


def _output_dir() -> Path:
    return Path(settings.BASE_DIR) / "Weather_Forcast_App" / "output"


def _safe_join_output(filename: str) -> Path:
    base = _output_dir().resolve()
    p = (base / filename).resolve()
    if base not in p.parents and p != base:
        raise Http404("Invalid path")
    if not p.exists() or not p.is_file():
        raise Http404("File not found")
    return p


def datasets_view(request):
    out = _output_dir()
    out.mkdir(parents=True, exist_ok=True)

    patterns = ["*.xlsx", "*.csv", "*.json", "*.txt"]
    files = []
    for pat in patterns:
        files.extend(out.glob(pat))

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    items = []
    for p in files:
        st = p.stat()
        items.append({
            "name": p.name,
            "size_mb": round(st.st_size / (1024 * 1024), 2),
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })

    latest = items[0] if items else None

    return render(request, "weather/Datasets.html", {
        "items": items,
        "latest": latest,
    })


def dataset_download_view(request, filename: str):
    p = _safe_join_output(filename)
    content_type, _ = mimetypes.guess_type(str(p))
    resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
    resp["Content-Disposition"] = f'attachment; filename="{p.name}"'
    return resp


def dataset_view_view(request, filename: str):
    p = _safe_join_output(filename)
    content_type, _ = mimetypes.guess_type(str(p))
    resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
    resp["Content-Disposition"] = f'inline; filename="{p.name}"'
    return resp
