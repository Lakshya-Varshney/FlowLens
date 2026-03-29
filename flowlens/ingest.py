"""
ingest.py — Git repository and CI log ingestion for FlowLens.

Speed strategy for large repos like kubernetes:
  FIRST RUN  → bare + blobless + shallow clone
               skips vendor/, staging/, all file contents
               kubernetes: ~2 min instead of 30 min
  RE-RUNS    → git fetch --depth=1 (only new commits since last run)
               typically < 10 seconds

Progress bars shown at every stage using tqdm.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Generator

import pandas as pd
from git import Repo, GitCommandError, InvalidGitRepositoryError
from tqdm import tqdm

logger = logging.getLogger("flowlens.ingest")

# Persistent repo cache directory
_REPO_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "repos"

# Column schema for the raw commits DataFrame
RAW_COMMIT_COLUMNS = [
    "commit_hash", "author_email", "author_name",
    "timestamp_utc", "commit_date", "commit_hour", "commit_weekday",
    "insertions", "deletions", "files_changed",
    "commit_message", "is_merge", "repo_name",
]

# tqdm bar format used consistently across all steps
_BAR_FMT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
_BAR_FMT_NOCOUNT = "{desc} |{bar}| {n_fmt} {unit} [{elapsed}]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_repo(
    repo_source: str,
    since_date: datetime | None = None,
    batch_size: int = 5000,
    depth: int = 500,
) -> pd.DataFrame:
    """
    Main entry point for git ingestion.

    Args:
        repo_source: HTTPS GitHub URL or local filesystem path.
        since_date:  Only include commits on or after this date.
        batch_size:  Commits processed in batches of this size.
        depth:       Shallow clone depth (0 = full history).

    Returns:
        pd.DataFrame with columns defined by RAW_COMMIT_COLUMNS.
    """
    repo, repo_name, tmp_dir = _open_repo(repo_source, depth=depth)

    try:
        print()

        # ── Single fast git log call — extracts ALL commit data without
        # touching blob objects (no network round-trips for blobless clones).
        # Replaces the old scan+extract loop that took 37 minutes on kubernetes.
        with tqdm(
            total=1,
            desc="  ⛏  Extracting commits via git log",
            unit=" batch",
            colour="green",
            bar_format="{desc} … [{elapsed}]",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            records = _bulk_extract_commits(repo, repo_name, since_date, depth)
            pbar.update(1)

        if not records:
            logger.warning("No commits found for the given date range.")
            return pd.DataFrame(columns=RAW_COMMIT_COLUMNS)

        # Show progress while building the DataFrame
        with tqdm(
            total=len(records),
            desc="  📦 Building DataFrame",
            unit=" commits",
            colour="cyan",
            bar_format=_BAR_FMT,
            dynamic_ncols=True,
            file=sys.stdout,
        ) as pbar:
            # Process in batches so progress bar updates smoothly
            all_batches = []
            for i in range(0, len(records), batch_size):
                chunk = records[i:i + batch_size]
                all_batches.append(pd.DataFrame(chunk, columns=RAW_COMMIT_COLUMNS))
                pbar.update(len(chunk))

        df = pd.concat(all_batches, ignore_index=True)
        print()
        logger.info("✓ Ingested %d commits from '%s'.", len(df), repo_name)
        return df

    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_ci_logs(ci_logs_dir: str) -> pd.DataFrame:
    """Parse JUnit XML test result files from a directory (Cross-Track B)."""
    ci_dir = Path(ci_logs_dir)
    xml_files = list(ci_dir.rglob("*.xml"))

    if not xml_files:
        logger.warning("No JUnit XML files found in %s", ci_logs_dir)
        return pd.DataFrame()

    records = []
    with tqdm(
        xml_files,
        desc="  📋 Parsing CI logs",
        unit=" files",
        colour="yellow",
        bar_format=_BAR_FMT,
        dynamic_ncols=True,
        file=sys.stdout,
    ) as pbar:
        for f in pbar:
            r = _parse_single_junit_xml(f)
            if r:
                records.append(r)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["run_date"] = pd.to_datetime(df["run_date"]).dt.date.astype(str)
    agg = df.groupby("run_date").agg(
        total_tests=("total_tests", "sum"),
        failed_tests=("failed_tests", "sum"),
        build_time_seconds=("build_time_seconds", "sum"),
    ).reset_index()
    agg["test_failure_density"] = agg["failed_tests"] / agg["total_tests"].replace(0, 1)
    agg["build_time_minutes"] = agg["build_time_seconds"] / 60
    return agg


# ---------------------------------------------------------------------------
# Repo opening
# ---------------------------------------------------------------------------

def _open_repo(repo_source: str, depth: int = 500) -> tuple[Repo, str, str | None]:
    """
    Open a repo from URL or local path.
    Remote URLs use a persistent bare+blobless cache for speed.
    """
    if not (repo_source.startswith("http://") or repo_source.startswith("https://")):
        local_path = Path(repo_source).resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Local repo path not found: {local_path}")
        try:
            repo = Repo(str(local_path))
        except InvalidGitRepositoryError as exc:
            raise ValueError(f"Not a valid git repository: {local_path}") from exc
        return repo, local_path.name, None

    repo_name = repo_source.rstrip("/").split("/")[-1].replace(".git", "")
    _REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = _REPO_CACHE_DIR / f"{repo_name}.git"

    if cache_dir.exists():
        return _open_cached_repo(repo_source, repo_name, cache_dir, depth)

    return _fresh_clone(repo_source, repo_name, cache_dir, depth)


def _open_cached_repo(
    repo_source: str,
    repo_name: str,
    cache_dir: Path,
    depth: int,
) -> tuple[Repo, str, None]:
    """Open cached repo and fetch only new commits."""
    logger.info("✓ Repo '%s' found in cache — fetching new commits only …", repo_name)

    try:
        # Spinner while fetching (fetch duration is unpredictable)
        # Use the same depth as the original clone so re-fetches cover the
        # same lookback window.  --depth=1 would only grab the very tip of
        # the branch and would cause --since to return 0 commits on the next run.
        fetch_depth = depth if depth > 0 else 500
        _run_with_spinner(
            cmd=[
                "git", "--git-dir", str(cache_dir),
                "fetch", "origin",
                f"--depth={fetch_depth}",
                "--no-tags",
                "--update-shallow",
            ],
            desc="  🔄 Fetching new commits",
            timeout=300,
        )
        logger.info("✓ Fetch complete.")
        return Repo(str(cache_dir)), repo_name, None

    except Exception as exc:
        logger.warning("Fetch failed (%s) — re-cloning from scratch …", exc)
        _safe_rmtree(cache_dir)
        return _fresh_clone(repo_source, repo_name, cache_dir, depth)


def _fresh_clone(
    repo_source: str,
    repo_name: str,
    cache_dir: Path,
    depth: int,
) -> tuple[Repo, str, None]:
    """
    Fresh bare + blobless clone with live progress bar.

    Why --bare + --filter=blob:none:
      --bare:             no working tree → skips vendor/(~300MB), staging/, third_party/
      --filter=blob:none: skips all file content blobs (we only need commit metadata)
      --no-tags:          skips hundreds of version tags
      --single-branch:    only default branch
      --depth N:          only last N commits
    """
    clone_depth = depth if depth > 0 else None

    cmd = [
        "git", "clone",
        "--bare",
        "--filter=blob:none",
        "--no-tags",
        "--single-branch",
        "--progress",         # makes git print progress to stderr
    ]
    if clone_depth:
        cmd += [f"--depth={clone_depth}"]
    cmd += [repo_source, str(cache_dir)]

    logger.info(
        "Cloning '%s' (bare+blobless, depth=%s) …",
        repo_name, clone_depth or "full"
    )

    _run_clone_with_progress(cmd, repo_name, cache_dir)

    logger.info("✓ Clone complete for '%s'.", repo_name)
    return Repo(str(cache_dir)), repo_name, None


# ---------------------------------------------------------------------------
# Progress display helpers
# ---------------------------------------------------------------------------

def _safe_rmtree(path: Path) -> bool:
    """
    Reliably delete a directory on Windows + OneDrive.
    OneDrive can hold file locks briefly, so we retry with a delay.
    Returns True if successfully deleted, False otherwise.
    """
    import time
    for attempt in range(5):
        try:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=False)
            return True
        except Exception as exc:
            if attempt < 4:
                logger.debug("rmtree attempt %d failed (%s) — retrying …", attempt + 1, exc)
                time.sleep(0.5)
            else:
                logger.warning("Could not delete '%s': %s", path, exc)
                return False
    return False


def _run_clone_with_progress(cmd: list[str], repo_name: str, cache_dir: Path) -> None:
    """
    Run git clone and show a live progress bar by parsing git's stderr output.

    Git --progress prints lines like:
      Counting objects:  42% (210/500)
      Receiving objects:  67% (335/500), 1.23 MiB | 2.34 MiB/s
    We parse these to drive a tqdm bar.
    """
    # Guard: if cache_dir exists from a previous broken clone, delete it first
    # so git doesn't refuse to clone into a non-empty directory
    if cache_dir.exists():
        logger.info("Removing broken cache directory before re-cloning …")
        _safe_rmtree(cache_dir)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Phases git reports in order
    phases = ["Counting objects", "Compressing objects", "Receiving objects", "Resolving deltas"]
    current_phase = ""

    bar = tqdm(
        total=100,
        desc=f"  📦 Cloning {repo_name}",
        unit="%",
        colour="cyan",
        bar_format=_BAR_FMT,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    last_pct = 0
    # Collect all stderr lines so we have them on failure
    stderr_lines: list[str] = []

    try:
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            stderr_lines.append(line)  # collect for error reporting

            for phase in phases:
                if phase in line:
                    if phase != current_phase:
                        current_phase = phase
                        bar.set_description(f"  📦 {phase}")
                        bar.reset()
                        last_pct = 0

                    pct_match = re.search(r"(\d+)%", line)
                    if pct_match:
                        pct = int(pct_match.group(1))
                        if pct > last_pct:
                            bar.update(pct - last_pct)
                            last_pct = pct
                    break

        process.wait(timeout=300)

    except subprocess.TimeoutExpired:
        process.kill()
        bar.close()
        _safe_rmtree(cache_dir)
        raise RuntimeError("Git clone timed out after 5 minutes.")
    finally:
        bar.update(100 - last_pct)
        bar.close()
        print()

    if process.returncode != 0:
        # Use the collected stderr lines — don't try to re-read the pipe
        error_msg = "\n".join(stderr_lines[-5:]) or "unknown git error"
        _safe_rmtree(cache_dir)
        raise RuntimeError(f"Failed to clone repo: {error_msg}")


def _run_with_spinner(cmd: list[str], desc: str, timeout: int = 120) -> None:
    """
    Run a subprocess and show an animated spinner while it runs.
    Used for operations where we can't easily parse progress (e.g. git fetch).
    """
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    stop_event = threading.Event()

    def _spin():
        i = 0
        while not stop_event.is_set():
            sys.stdout.write(f"\r  {spinner_chars[i % len(spinner_chars)]}  {desc} …   ")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write(f"\r  ✓  {desc} — done          \n")
        sys.stdout.flush()

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stop_event.set()
        spin_thread.join()

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

    except subprocess.TimeoutExpired:
        stop_event.set()
        spin_thread.join()
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd[:3])}")

    except Exception:
        stop_event.set()
        spin_thread.join()
        raise


# ---------------------------------------------------------------------------
# Commit iteration — uses fast bulk git log instead of per-commit API calls
# ---------------------------------------------------------------------------

def _iter_commits(repo: Repo, since_date: datetime | None) -> Generator:
    """
    Yield commits using git's native --since flag (much faster than Python filtering).
    NOTE: This is only used for commit counting. Actual extraction uses _bulk_extract.
    """
    try:
        rev = _resolve_head_ref(repo)
        kwargs: dict = {}
        if since_date:
            kwargs["since"] = since_date.strftime("%Y-%m-%d")

        for commit in repo.iter_commits(rev, **kwargs):
            yield commit

    except GitCommandError as exc:
        logger.warning("Error iterating commits: %s", exc)
        return


def _bulk_extract_commits(
    repo: Repo,
    repo_name: str,
    since_date: datetime | None,
    depth: int,
) -> list[dict]:
    """
    Extract ALL commit data in a single git log --numstat call.

    This replaces the old loop that called commit.stats on every commit,
    which triggered a network round-trip per commit for blobless clones.

    One git log call → parse in Python → done.
    Speed: 2198 commits in ~3 seconds instead of 37 minutes.

    Format used (fields separated by \x00, records by \x1e):
      \x1e hash \x00 author_email \x00 author_name \x00 timestamp \x00 parents \x00 message
      [blank line]
      insertions TAB deletions TAB filename   (one per changed file, from --numstat)

    NOTE: \x1e is placed at the START of each record (not the end) so it acts
    as a true record START sentinel — this avoids off-by-one parsing issues when
    git omits the trailing newline on the very last record.
    """
    git_dir = str(repo.git_dir)

    # %P = parent hashes (space-separated) — used for merge detection
    # \x1e placed at record START so splitting on it gives clean per-record chunks
    fmt = "%x1e%H%x00%ae%x00%an%x00%ct%x00%P%x00%s"

    cmd = [
        "git",
        "--git-dir", git_dir,
        "log",
        f"--format={fmt}",
        "--numstat",             # insertions deletions filename (no blob fetch)
        "--no-merges",           # skip merge commits for numstat (they're noisy)
    ]

    if since_date:
        cmd += [f"--since={since_date.strftime('%Y-%m-%d')}"]

    # NOTE: Do NOT add -n/--max-count here. When combined with --since on a
    # shallow clone, -n caps results before the date filter is applied, which
    # causes git to return 0 commits even when matching commits exist.
    # The clone depth (set at clone time) already limits how far back we look.

    # Resolve the ref to log from — try standard names first, then any remote ref.
    # For bare repos (kubernetes etc.) FETCH_HEAD is the most reliable pointer.
    rev = None
    try:
        rev = _resolve_head_ref(repo)
    except Exception:
        pass  # git log without an explicit ref still works for most repos

    if rev:
        cmd += [rev]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300,          # 5 minute hard limit (kubernetes can be slow)
            cwd=git_dir,
        )
    except subprocess.TimeoutExpired:
        logger.warning("git log timed out after 5 minutes — falling back to GitPython iteration.")
        return _gitpython_fallback(repo, repo_name, since_date, depth)

    if result.returncode != 0:
        stderr_msg = result.stderr.decode("utf-8", errors="replace")[:300]
        logger.warning(
            "git log failed (rc=%d): %s — falling back to GitPython.",
            result.returncode,
            stderr_msg,
        )
        return _gitpython_fallback(repo, repo_name, since_date, depth)

    raw = result.stdout.decode("utf-8", errors="replace")

    if not raw.strip():
        # Empty output: shallow clone may not contain commits in the date range.
        # Try without --since to confirm the clone has any commits at all.
        logger.warning(
            "git log returned empty output with --since=%s. "
            "The shallow clone (depth=%d) may not reach that date. "
            "Retrying without date filter to check clone health …",
            since_date.strftime("%Y-%m-%d") if since_date else "N/A",
            depth,
        )
        return _gitpython_fallback(repo, repo_name, since_date, depth)

    return _parse_git_log_output(raw, repo_name)


def _parse_git_log_output(raw: str, repo_name: str) -> list[dict]:
    """
    Parse the output of git log --format=... --numstat into a list of dicts.

    \x1e is placed at the START of each record (see _bulk_extract_commits),
    so splitting on it produces one empty leading chunk followed by one chunk
    per commit.

    Within each record:
      - First non-empty line: format fields (hash, email, name, timestamp, parents, subject)
      - Remaining lines: numstat lines (insertions TAB deletions TAB filename)
    """
    records: list[dict] = []

    # Split on record-start sentinel — first element is always empty, skip it.
    for raw_record in raw.split("\x1e"):
        raw_record = raw_record.strip()
        if not raw_record:
            continue

        lines = raw_record.splitlines()
        if not lines:
            continue

        # Find first non-empty line as header; numstat starts on the next line.
        header = ""
        numstat_start = 1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                header = stripped
                numstat_start = i + 1
                break

        parts = header.split("\x00")
        if len(parts) < 6:
            continue

        commit_hash, author_email, author_name, timestamp_str, parents_str, subject = parts[:6]

        try:
            timestamp = int(timestamp_str)
        except ValueError:
            continue

        committed_dt = datetime.utcfromtimestamp(timestamp)
        is_merge = int(len(parents_str.strip().split()) > 1)

        # Parse numstat lines
        total_insertions = 0
        total_deletions = 0
        files_changed: list[str] = []

        for line in lines[numstat_start:]:
            line = line.strip()
            if not line:
                continue
            parts_stat = line.split("\t")
            if len(parts_stat) < 3:
                continue
            ins_str, del_str, filename = parts_stat[0], parts_stat[1], parts_stat[2]
            # Binary files show "-" instead of a number
            try:
                total_insertions += int(ins_str)
            except ValueError:
                pass
            try:
                total_deletions += int(del_str)
            except ValueError:
                pass
            if filename:
                files_changed.append(filename)

        raw_email = author_email.strip().lower()
        if not raw_email:
            raw_email = _email_fallback(author_name or "unknown")

        records.append({
            "commit_hash": commit_hash.strip(),
            "author_email": raw_email,
            "author_name": (author_name or "unknown").strip(),
            "timestamp_utc": timestamp,
            "commit_date": committed_dt.strftime("%Y-%m-%d"),
            "commit_hour": committed_dt.hour,
            "commit_weekday": committed_dt.weekday(),
            "insertions": total_insertions,
            "deletions": total_deletions,
            "files_changed": files_changed,
            "commit_message": subject.strip()[:500],
            "is_merge": is_merge,
            "repo_name": repo_name,
        })

    return records


def _gitpython_fallback(
    repo: Repo,
    repo_name: str,
    since_date,
    depth: int,
) -> list[dict]:
    """
    Fallback commit extraction using GitPython iteration.

    Used when the fast git log subprocess path fails or returns empty results.
    This is slower but more robust across git versions and repo configurations.

    NOTE: commit.stats is NOT called here to avoid per-commit blob fetches
    on blobless clones. insertions/deletions will be 0 for all commits.
    """
    logger.info("GitPython fallback: iterating commits for '%s' …", repo_name)
    records: list[dict] = []

    try:
        rev = _resolve_head_ref(repo)
    except Exception:
        rev = None

    iter_kwargs: dict = {}
    if since_date:
        iter_kwargs["since"] = since_date.strftime("%Y-%m-%d")
    if depth > 0:
        iter_kwargs["max_count"] = depth

    try:
        commits_iter = (
            repo.iter_commits(rev, **iter_kwargs)
            if rev
            else repo.iter_commits(**iter_kwargs)
        )
        for commit in commits_iter:
            records.append(_extract_commit_record(commit, repo_name))
    except GitCommandError as exc:
        logger.error("GitPython fallback also failed: %s", exc)

    if not records:
        # Last resort: try without since_date in case clone is too shallow
        logger.warning(
            "GitPython fallback with --since returned 0 commits. "
            "Retrying without date filter (clone may be shallower than the lookback window)."
        )
        no_date_kwargs: dict = {}
        if depth > 0:
            no_date_kwargs["max_count"] = depth
        try:
            commits_iter = (
                repo.iter_commits(rev, **no_date_kwargs)
                if rev
                else repo.iter_commits(**no_date_kwargs)
            )
            for commit in commits_iter:
                records.append(_extract_commit_record(commit, repo_name))
        except GitCommandError as exc:
            logger.error("GitPython fallback (no date) also failed: %s", exc)

    logger.info("GitPython fallback: extracted %d commits.", len(records))
    return records


def _resolve_head_ref(repo: Repo) -> str:
    """Resolve the correct HEAD ref for both bare and non-bare repos."""
    for ref in ("HEAD", "main", "master", "trunk", "develop", "FETCH_HEAD"):
        try:
            repo.commit(ref)
            return ref
        except Exception:
            continue

    try:
        remote_refs = [r.name for r in repo.references if "origin/" in r.name]
        if remote_refs:
            return remote_refs[0]
    except Exception:
        pass

    raise GitCommandError("git rev-parse HEAD", 128, "Cannot resolve HEAD ref.")


# ---------------------------------------------------------------------------
# Commit record extraction (fallback for non-bulk path)
# ---------------------------------------------------------------------------

def _extract_commit_record(commit, repo_name: str) -> dict:
    """
    Extract a flat dict from a GitPython commit object.
    NOTE: commit.stats is intentionally NOT called here — it triggers
    per-commit blob fetches which are extremely slow on blobless clones.
    insertions/deletions default to 0; the bulk path (_bulk_extract_commits)
    gets accurate values via --numstat without any blob fetches.
    """
    committed_dt = datetime.utcfromtimestamp(commit.committed_date)

    raw_email = (commit.author.email or "").strip().lower()
    if not raw_email:
        raw_email = _email_fallback(commit.author.name or "unknown")

    return {
        "commit_hash": commit.hexsha,
        "author_email": raw_email,
        "author_name": (commit.author.name or "unknown").strip(),
        "timestamp_utc": int(commit.committed_date),
        "commit_date": committed_dt.strftime("%Y-%m-%d"),
        "commit_hour": committed_dt.hour,
        "commit_weekday": committed_dt.weekday(),
        "insertions": 0,       # no blob fetch — use bulk path for accurate values
        "deletions": 0,
        "files_changed": [],
        "commit_message": (commit.message or "").strip()[:500],
        "is_merge": int(len(commit.parents) > 1),
        "repo_name": repo_name,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _email_fallback(author_name: str) -> str:
    h = hashlib.md5(author_name.encode()).hexdigest()[:8]
    return f"{h}@unknown.local"


def _parse_single_junit_xml(xml_path: Path) -> dict | None:
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
        total = sum(int(s.attrib.get("tests", 0)) for s in suites)
        failed = sum(
            int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0))
            for s in suites
        )
        time_s = sum(float(s.attrib.get("time", 0)) for s in suites)
        run_date = _infer_date_from_filename(xml_path) or datetime.utcnow().strftime("%Y-%m-%d")
        return {
            "run_date": run_date,
            "total_tests": total,
            "failed_tests": failed,
            "build_time_seconds": time_s,
            "source_file": xml_path.name,
        }
    except ET.ParseError:
        return None


def _infer_date_from_filename(xml_path: Path) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", xml_path.stem)
    return match.group(1) if match else None