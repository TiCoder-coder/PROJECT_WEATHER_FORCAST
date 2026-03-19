"""
crawl_queue.py — Shared Crawl Job Queue
========================================

Module này cung cấp một singleton job queue dùng chung cho tất cả
các crawl views. Thay vì mỗi view tự spawn thread và có thể chạy
đồng thời nhiều crawl gây quá tải server, tất cả đều submit job vào
queue và worker thread xử lý tuần tự (MAX_WORKERS = 1).

Lợi ích:
  - Tránh nhiều crawl chạy đồng thời → không làm down server
  - Queue job thay vì reject → người dùng không bị mất yêu cầu
  - Centralized logging và trạng thái cho tất cả loại crawl
  - Frontend nhận job_id → poll đúng kết quả của request mình

Usage từ view:
    from Weather_Forcast_App.crawl_queue import get_queue

    q = get_queue()
    job = q.enqueue(
        script_path="/abs/path/script.py",
        output_dir="/abs/path/output",
        label="Vrain API",
    )
    # Trả về CrawlJob với: job.job_id, job.status, queue_position(job.job_id)
"""

import os
import sys
import queue
import threading
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# CẤU HÌNH
# ============================================================

# MAX_WORKERS: số worker xử lý queue đồng thời.
# Để 1 để đảm bảo không bao giờ có 2 crawl chạy cùng lúc.
MAX_WORKERS = 1

# LOG_LIMIT: giới hạn số dòng log lưu trong RAM mỗi job.
LOG_LIMIT = 3000

# JOB_HISTORY: số job đã hoàn thành (done/failed) giữ lại trong bộ nhớ.
# Job cũ hơn sẽ bị xóa để tránh memory leak.
JOB_HISTORY = 50


# ============================================================
# JOB STATUS
# ============================================================

class JobStatus:
    QUEUED  = "queued"   # đang chờ trong hàng đợi
    RUNNING = "running"  # đang thực thi
    DONE    = "done"     # hoàn thành thành công
    FAILED  = "failed"   # kết thúc với lỗi


# ============================================================
# CrawlJob: MỘT ĐƠN VỊ CÔNG VIỆC CRAWL
# ============================================================

@dataclass
class CrawlJob:
    """
    Đại diện cho một công việc crawl dữ liệu.

    Attributes:
        job_id:       ID duy nhất (uuid4 hex) để frontend track
        script_path:  Đường dẫn tuyệt đối tới script Python cần chạy
        output_dir:   Thư mục output của script
        label:        Tên hiển thị (vd: "Vrain API", "Selenium")
        extra_args:   Args bổ sung truyền vào script
        status:       Trạng thái hiện tại (JobStatus.*)
        logs:         Danh sách dòng log (realtime streaming)
        returncode:   Exit code của subprocess sau khi xong
        created_at:   Thời điểm enqueue
        started_at:   Thời điểm bắt đầu chạy
        finished_at:  Thời điểm kết thúc
        error:        Mô tả lỗi nếu có exception
        last_file:    Tên file output mới nhất sau khi chạy
        last_size_mb: Kích thước file output (MB)
    """
    job_id:       str
    script_path:  str
    output_dir:   str
    label:        str             = ""
    extra_args:   list            = field(default_factory=list)
    status:       str             = JobStatus.QUEUED
    logs:         list            = field(default_factory=list)
    returncode:   Optional[int]   = None
    created_at:   str             = ""
    started_at:   Optional[str]   = None
    finished_at:  Optional[str]   = None
    error:        Optional[str]   = None
    last_file:    Optional[str]   = None
    last_size_mb: Optional[float] = None

    def is_active(self) -> bool:
        return self.status in (JobStatus.QUEUED, JobStatus.RUNNING)

    def to_dict(self) -> dict:
        return {
            "job_id":       self.job_id,
            "label":        self.label,
            "status":       self.status,
            "returncode":   self.returncode,
            "created_at":   self.created_at,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "last_file":    self.last_file,
            "last_size_mb": self.last_size_mb,
            "log_count":    len(self.logs),
        }


# ============================================================
# CrawlJobQueue: SINGLETON QUEUE MANAGER
# ============================================================

class CrawlJobQueue:
    """
    Quản lý hàng đợi các crawl job.

    Khi khởi tạo, sẽ tự động spawn MAX_WORKERS daemon threads để
    xử lý job từ queue một cách tuần tự.
    """

    def __init__(self, max_workers: int = MAX_WORKERS):
        self._q      = queue.Queue()           # FIFO queue chứa job_id
        self._jobs   = {}                      # job_id → CrawlJob registry
        self._lock   = threading.Lock()        # bảo vệ _jobs
        self._max_w  = max_workers

        # Khởi động worker daemon threads
        for _ in range(max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()

    # ----------------------------------------------------------
    # PUBLIC: enqueue
    # ----------------------------------------------------------

    def enqueue(
        self,
        script_path,
        output_dir,
        label: str = "",
        extra_args: list = None,
    ) -> CrawlJob:
        """
        Tạo CrawlJob mới và đẩy vào hàng đợi.

        Returns:
            CrawlJob object (check .job_id để lưu ở frontend)
        """
        job_id = uuid.uuid4().hex
        job = CrawlJob(
            job_id      = job_id,
            script_path = str(script_path),
            output_dir  = str(output_dir),
            label       = label,
            extra_args  = list(extra_args or []),
            status      = JobStatus.QUEUED,
            created_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._trim_history()

        self._q.put(job_id)
        return job

    # ----------------------------------------------------------
    # PUBLIC: get_job
    # ----------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[CrawlJob]:
        """Lấy CrawlJob theo job_id. Trả về None nếu không tìm thấy."""
        with self._lock:
            return self._jobs.get(job_id)

    # ----------------------------------------------------------
    # PUBLIC: queue_position
    # ----------------------------------------------------------

    def queue_position(self, job_id: str) -> int:
        """
        Trả về vị trí 1-based trong hàng đợi.
        0 → đang chạy hoặc đã xong (không còn trong queue).
        """
        items = list(self._q.queue)
        try:
            return items.index(job_id) + 1
        except ValueError:
            return 0

    # ----------------------------------------------------------
    # PUBLIC: queue_length
    # ----------------------------------------------------------

    def queue_length(self) -> int:
        """Số job đang chờ trong hàng đợi (chưa bắt đầu chạy)."""
        return self._q.qsize()

    # ----------------------------------------------------------
    # PUBLIC: running_jobs
    # ----------------------------------------------------------

    def running_jobs(self) -> list:
        """Danh sách các CrawlJob đang ở trạng thái RUNNING."""
        with self._lock:
            return [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]

    # ----------------------------------------------------------
    # PUBLIC: latest_job_for
    # ----------------------------------------------------------

    def latest_job_for(self, label_keyword: str = "") -> Optional[CrawlJob]:
        """
        Lấy CrawlJob gần nhất match label_keyword (case-insensitive substring).
        Ưu tiên job đang active; fallback về job cuối theo created_at.
        Dùng để render trang GET khi chưa có job_id cụ thể.
        """
        with self._lock:
            candidates = [
                j for j in self._jobs.values()
                if (not label_keyword) or (label_keyword.lower() in j.label.lower())
            ]
        if not candidates:
            return None
        active = [j for j in candidates if j.is_active()]
        if active:
            return active[-1]
        return sorted(candidates, key=lambda j: j.created_at)[-1]

    # ----------------------------------------------------------
    # PUBLIC: get_logs_since
    # ----------------------------------------------------------

    def get_logs_since(self, job_id: str, since: int = 0):
        """
        Trả về (new_lines, next_since) cho polling incremental.

        Args:
            job_id: ID của job cần lấy log
            since:  Offset đã nhận từ trước (0 = lấy từ đầu)

        Returns:
            (list[str], int) — dòng log mới, con trỏ mới
        """
        job = self.get_job(job_id)
        if job is None:
            return [], since
        with self._lock:
            logs = list(job.logs)
        return logs[since:], len(logs)

    # ----------------------------------------------------------
    # PUBLIC: get_status_dict
    # ----------------------------------------------------------

    def get_status_dict(self, job_id: str, since: int = 0) -> dict:
        """
        Tạo dict đầy đủ trả về cho tail/polling endpoints.
        Tương thích với format mà JS frontend đang expect.
        """
        job = self.get_job(job_id)
        if job is None:
            return {
                "ok":         False,
                "error":      "Job not found",
                "is_running": False,
                "is_queued":  False,
            }
        lines, next_since = self.get_logs_since(job_id, since)
        pos = self.queue_position(job_id)
        return {
            "ok":             True,
            "job_id":         job.job_id,
            "status":         job.status,
            "is_running":     job.status == JobStatus.RUNNING,
            "is_queued":      job.status == JobStatus.QUEUED,
            "queue_position": pos,
            "queue_total":    self.queue_length(),
            "lines":          lines,
            "next_since":     next_since,
            "returncode":     job.returncode,
            "last_crawl_time":job.finished_at,
            "last_file":      job.last_file,
            "last_size_mb":   job.last_size_mb,
            "label":          job.label,
        }

    # ----------------------------------------------------------
    # PUBLIC: get_queue_info
    # ----------------------------------------------------------

    def get_queue_info(self) -> dict:
        """Trả về tổng quan hàng đợi để hiển thị status."""
        with self._lock:
            jobs_list = [j.to_dict() for j in self._jobs.values()]
        running = [j for j in jobs_list if j["status"] == JobStatus.RUNNING]
        queued  = [j for j in jobs_list if j["status"] == JobStatus.QUEUED]
        return {
            "queue_length":   self.queue_length(),
            "running_count":  len(running),
            "running_jobs":   running,
            "queued_jobs":    queued,
            "total_tracked":  len(jobs_list),
        }

    # ----------------------------------------------------------
    # PRIVATE: _trim_history
    # ----------------------------------------------------------

    def _trim_history(self):
        """
        Xóa bớt job cũ đã hoàn thành để tránh memory leak.
        Phải được gọi trong khi đang giữ self._lock.
        """
        done_ids = [
            jid for jid, j in self._jobs.items()
            if j.status in (JobStatus.DONE, JobStatus.FAILED)
        ]
        if len(done_ids) > JOB_HISTORY:
            for jid in done_ids[: len(done_ids) - JOB_HISTORY]:
                del self._jobs[jid]

    # ----------------------------------------------------------
    # PRIVATE: _push_log
    # ----------------------------------------------------------

    def _push_log(self, job: CrawlJob, line: str):
        """Thread-safe append log line vào job, có giới hạn LOG_LIMIT."""
        line = (line or "").rstrip("\n")
        if not line:
            return
        with self._lock:
            job.logs.append(line)
            if len(job.logs) > LOG_LIMIT:
                job.logs = job.logs[-LOG_LIMIT:]

    # ----------------------------------------------------------
    # PRIVATE: _worker_loop
    # ----------------------------------------------------------

    def _worker_loop(self):
        """Vòng lặp vô hạn trong daemon thread — liên tục lấy và xử lý job."""
        while True:
            job_id = self._q.get()
            job = self.get_job(job_id)
            if job is not None:
                self._run_job(job)
            self._q.task_done()

    # ----------------------------------------------------------
    # PRIVATE: _run_job
    # ----------------------------------------------------------

    def _run_job(self, job: CrawlJob):
        """
        Thực thi một CrawlJob:
          1. Set status → RUNNING
          2. Spawn subprocess chạy script
          3. Stream stdout → push log
          4. Cập nhật status → DONE / FAILED
          5. Scan file output mới nhất
        """
        # Cập nhật trạng thái → đang chạy
        with self._lock:
            job.status     = JobStatus.RUNNING
            job.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            output_dir  = Path(job.output_dir)
            script_path = Path(job.script_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            self._push_log(job, f"========== START: {job.label} ==========")
            self._push_log(job, f"Script  : {script_path}")
            self._push_log(job, f"Output  : {output_dir}")
            self._push_log(job, f"Started : {job.started_at}")

            if not script_path.exists():
                self._push_log(job, "[ERROR] Script không tồn tại!")
                with self._lock:
                    job.returncode  = -1
                    job.status      = JobStatus.FAILED
                    job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return

            # Môi trường unbuffered để log realtime
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"]  = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["CRAWL_MODE"]       = "once"

            cmd = [sys.executable, "-u", str(script_path)] + job.extra_args
            self._push_log(job, f"CMD     : {' '.join(cmd)}")

            # cwd = thư mục cha của thư mục scripts (= APP_ROOT)
            app_root = str(script_path.parent.parent)

            proc = subprocess.Popen(
                cmd,
                cwd      = app_root,
                stdout   = subprocess.PIPE,
                stderr   = subprocess.STDOUT,
                text     = True,
                bufsize  = 1,
                env      = env,
                encoding = "utf-8",
                errors   = "replace",
            )

            # Stream log từ stdout theo từng dòng
            for line in proc.stdout:
                self._push_log(job, line)

            rc = proc.wait()

            with self._lock:
                job.returncode  = rc
                job.status      = JobStatus.DONE if rc == 0 else JobStatus.FAILED
                job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Scan file output mới nhất
            try:
                exts  = {".xlsx", ".csv", ".xls"}
                files = [
                    p for p in output_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts
                ]
                if files:
                    latest = max(files, key=lambda p: p.stat().st_mtime)
                    with self._lock:
                        job.last_file    = latest.name
                        job.last_size_mb = round(latest.stat().st_size / (1024 * 1024), 2)
            except Exception:
                pass

            self._push_log(job, f"========== DONE (rc={rc}) ==========")
            if job.last_file:
                self._push_log(job, f"Output  : {job.last_file} ({job.last_size_mb} MB)")

        except Exception as e:
            with self._lock:
                job.returncode  = -1
                job.status      = JobStatus.FAILED
                job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                job.error       = repr(e)
            self._push_log(job, f"[EXCEPTION] {repr(e)}")


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_instance:      Optional[CrawlJobQueue] = None
_instance_lock: threading.Lock          = threading.Lock()


def get_queue() -> CrawlJobQueue:
    """
    Trả về singleton instance của CrawlJobQueue.
    Thread-safe, lazy initialization.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CrawlJobQueue(max_workers=MAX_WORKERS)
    return _instance
