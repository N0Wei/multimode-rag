"""Convert an uploaded PDF to the complete MinerU archive expected by ingestion."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _mineru_kit_executable() -> str:
    if sys.platform == "win32":
        candidate = Path(sys.executable).parent / "Scripts" / "mineru-kit.exe"
    else:
        candidate = Path(sys.executable).parent / "mineru-kit"
    executable = candidate if candidate.is_file() else shutil.which("mineru-kit")
    if not executable:
        raise RuntimeError("未找到 mineru-kit；请在已安装 MinerU 的 rag-agent 环境中启动项目")
    return str(executable)


def parse_pdf_with_mineru(pdf: Path, archive: Path, *, page_range: str = "all",
                          timeout_seconds: int = 3600) -> Path:
    """Run local Flash+TXT and verify its full ZIP before ingestion.

    The source PDF is kept separately; Markdown alone lacks the image assets
    and block geometry required by the current parent/child parser.
    """
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("MinerU 输入必须是 PDF 文件")
    archive.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _mineru_kit_executable(), "parse", str(pdf.resolve()),
        "--output", str(archive.resolve()), "--format", "zip",
        "--tier", "flash", "--ocr-mode", "txt", "--pages", page_range,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MinerU 解析超过 {timeout_seconds} 秒，活动索引未切换") from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "未知错误").strip()[-1200:]
        raise RuntimeError(f"MinerU 解析失败：{details}")
    if not zipfile.is_zipfile(archive):
        raise ValueError("MinerU 未生成有效 ZIP，活动索引未切换")
    with zipfile.ZipFile(archive) as result:
        names = result.namelist()
        middle = [name for name in names if name.endswith("middle_json.json")]
        if len(middle) != 1:
            raise ValueError("MinerU ZIP 缺少唯一的 middle_json.json")
        raw = json.loads(result.read(middle[0]))
        if not raw.get("pages"):
            raise ValueError("MinerU 未解析出任何页面")
    return archive
