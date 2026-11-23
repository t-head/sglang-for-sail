# SPDX-License-Identifier: Apache-2.0
"""Send one signed Markdown notification for a public CI report.

Nightly uses --nightly ARTIFACT_DIR. Other trusted workflows can pass --report
FILE with {"elapsed_s": 600, "results": [{"model": "example", "status": "pass"}]}.
Failure rows have status "fail" and may include phase, score, golden_score and
first_failing_sha. The report and artifacts must contain only public evidence.
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

URL = "PPU_CI_NOTIFY_WEBHOOK_URL"
SECRET = "PPU_CI_NOTIFY_SIGNING_SECRET"
ROSTER = "PPU_CI_NOTIFY_ROSTER_JSON"
MODEL = re.compile(r"[a-z0-9][a-z0-9_.-]{0,79}")
REPOSITORY = re.compile(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+")
REF = re.compile(r"[a-zA-Z0-9_./-]{1,128}")
SHA = re.compile(r"[0-9a-f]{40}")
PHASES = (
    "approval",
    "selection",
    "build",
    "startup",
    "gsm8k_score vs golden",
    "micro_e2e",
    "cleanup",
    "private_retention",
    "model_cache_check",
    "runtime_setup",
    "matrix",
    "missing evidence",
    "unknown",
)
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def warn(message):
    print(f"::warning::{message}", file=sys.stderr)


def json_file(path, limit=64 * 1024):
    if path.parent.is_symlink():
        raise ValueError("Invalid evidence directory")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("Invalid evidence file")
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Evidence file too large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def score(value):
    return value if type(value) in (int, float) and 0 <= value <= 1 else None


def signed_url(url, secret, timestamp_ms):
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.fragment
        or any(character.isspace() for character in url)
    ):
        raise ValueError("Expected an HTTPS webhook URL")
    if not secret:
        return url
    timestamp = str(timestamp_ms)
    digest = hmac.digest(
        secret.encode(), f"{timestamp}\n{secret}".encode(), hashlib.sha256
    )
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if key not in {"timestamp", "sign"}
    ]
    query.extend(
        (("timestamp", timestamp), ("sign", base64.b64encode(digest).decode()))
    )
    return parts._replace(query=urlencode(query)).geturl()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def deliver(url, secret, payload):
    try:
        request = Request(
            signed_url(url, secret, time.time_ns() // 1_000_000),
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with build_opener(NoRedirect()).open(request, timeout=8) as response:
            body = response.read(16_385)
        if len(body) > 16_384:
            raise ValueError("Response too large")
        if body.strip():
            result = json.loads(body)
            if not isinstance(result, dict) or result.get("errcode", 0) != 0:
                warn("Markdown webhook rejected the notification.")
                return False
    except HTTPError as error:
        warn(f"Markdown webhook returned HTTP {error.code}.")
        return False
    except (OSError, HTTPException, ValueError, RecursionError):
        # Exception details may include credentials or a server response.
        warn("Markdown webhook delivery failed.")
        return False
    return True


def recipients(raw, failed_models, weekday):
    if not raw:
        return [], []
    try:
        roster = json.loads(raw)
        if not isinstance(roster, dict):
            raise ValueError
        days = roster.get("weekdays", {})
        owners = roster.get("models", {})
        if not isinstance(days, dict) or not isinstance(owners, dict):
            raise ValueError
        mobiles, users = {}, {}
        for entry in [days.get(weekday, [])] + [
            owners.get(model, []) for model in failed_models
        ]:
            if not isinstance(entry, list):
                raise ValueError
            for target in entry:
                if not isinstance(target, dict):
                    raise ValueError
                mobile, user = target.get("mobile"), target.get("user_id")
                if isinstance(mobile, str) and re.fullmatch(r"\+?[0-9]{7,15}", mobile):
                    mobiles[mobile] = None
                elif isinstance(user, str) and re.fullmatch(
                    r"[A-Za-z0-9_.-]{1,64}", user
                ):
                    users[user] = None
                else:
                    raise ValueError
        return list(mobiles), list(users)
    except (ValueError, RecursionError):
        warn("Invalid notification roster. Sending without mentions.")
        return [], []


def public_context():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    ref = os.environ.get("NOTIFY_REF", os.environ.get("GITHUB_REF_NAME", ""))
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not REPOSITORY.fullmatch(repo) or not re.fullmatch(r"[0-9]+", run_id):
        raise ValueError("Missing public GitHub run identity")
    ref = ref if REF.fullmatch(ref) else "unknown"
    event = event if re.fullmatch(r"[a-z_]{1,32}", event) else "unknown"
    return repo, ref, event, run_id


def run_jobs(expected):
    """Resolve each expected job to the attempt that actually executed it."""
    repo, _, _, run_id = public_context()
    jobs, elapsed = {}, None
    try:
        attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
        if attempt < 1:
            raise ValueError
        route = f"repos/{repo}/actions/runs/{run_id}/attempts"

        def api(endpoint, *options):
            output = subprocess.check_output(
                ["gh", "api", "--hostname", "github.com", endpoint, *options],
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key not in (URL, SECRET, ROSTER)
                },
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            return json.loads(output)

        info = api(f"{route}/{attempt}")
        started = datetime.fromisoformat(info["run_started_at"].replace("Z", "+00:00"))
        elapsed = max(0, int(time.time() - started.timestamp()))
        if not expected:
            return jobs, elapsed

        executions = {}
        for number in range(1, attempt + 1):
            pages = api(f"{route}/{number}/jobs?per_page=100", "--paginate", "--slurp")
            seen = set()
            for page in pages:
                for job in page["jobs"]:
                    name = job["name"]
                    if name not in expected:
                        continue
                    if name in seen:
                        raise ValueError("Ambiguous job name")
                    seen.add(name)
                    # Partial reruns can copy prior jobs with new IDs and attempt
                    # numbers. Their original execution timestamps are retained.
                    timestamps = job["started_at"], job["completed_at"]
                    if executions.get(name) != timestamps:
                        executions[name] = timestamps
                        executed_in = number
                    else:
                        executed_in = jobs[name][1]
                    jobs[name] = job["conclusion"], executed_in
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
        # A later snapshot could contain a failure that supersedes an old pass.
        jobs.clear()
        warn("GitHub job metadata is incomplete.")
    return jobs, elapsed


def model_result(model, job, read_evidence):
    """Only a successful job with matching public evidence counts as a pass."""
    conclusion, attempt = job or (None, None)
    result = {"model": model, "status": "fail", "phase": "missing evidence"}
    if attempt is not None and conclusion in ("success", "failure", "timed_out"):
        try:
            status, phase, actual, golden = read_evidence(attempt)
        except (OSError, KeyError, TypeError, ValueError, RecursionError):
            return result
        if conclusion == "success" and status == "pass":
            result["status"] = "pass"
        elif status == "fail":
            result.update(phase=phase, score=actual, golden_score=golden)
        else:
            result["phase"] = "unknown"
    return result


def make_payload(report, roster, weekday):
    repo, ref, event, run_id = public_context()
    rows = report["results"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected one or more result rows")
    failures = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("model"), str)
            or not MODEL.fullmatch(row["model"])
            or row.get("status") not in ("pass", "fail")
        ):
            raise ValueError("Invalid result row")
        if row["status"] == "fail":
            failures.append(row)

    passed = len(rows) - len(failures)
    headline = (
        f"[❌ {len(failures)} 项失败 · 待认领]"
        if failures
        else f"[✅ 全绿 {passed}/{len(rows)}]"
    )
    title = f"{headline} {repo} / {ref} · {event}"
    elapsed = report.get("elapsed_s")
    duration = "未知"
    if type(elapsed) is int and elapsed >= 0:
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{hours}h {minutes:02d}m {seconds:02d}s"
    link = f"https://github.com/{repo}/actions/runs/{run_id}"
    text = [
        f"### {headline} `{repo} / {ref}` · {event}",
        f"通过 **{passed}/{len(rows)}** · 总耗时 {duration} · [Run]({link})",
    ]
    for row in failures:
        phase = row.get("phase", "unknown")
        phase = phase if phase in PHASES else "unknown"
        details = [phase]
        actual, golden = score(row.get("score")), score(row.get("golden_score"))
        if actual is not None:
            reference = f"{golden:.2%}" if golden is not None else "—"
            details.append(f"GSM8K {actual:.2%} vs {reference}")
        sha = row.get("first_failing_sha")
        if isinstance(sha, str) and SHA.fullmatch(sha):
            details.append(f"first_failing_sha `{sha}`")
        text.append(f"- `{row['model']}` · {' · '.join(details)} · [Run]({link})")
    mobiles, users = (
        recipients(roster, [row["model"] for row in failures], weekday)
        if failures
        else ([], [])
    )
    payload = {"msgtype": "markdown", "markdown": {"title": title}}
    if mobiles or users:
        text += [
            " ".join(f"@{target}" for target in mobiles + users),
            "请被 @ 同学于 10:30 前群内回复初判归因（机器环境 / 框架算子回归 / 权重配置）",
        ]
        payload["at"] = {"atMobiles": mobiles, "atUserIds": users, "isAtAll": False}
    payload["markdown"]["text"] = "\n\n".join(text)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--nightly", type=Path, metavar="ARTIFACT_DIR")
    mode.add_argument("--report", type=Path, metavar="FILE")
    mode.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    url = os.environ.get(URL, "").strip()
    if not url:
        if args.smoke_test:
            warn("Markdown webhook URL is not configured.")
            return 1
        return 0
    try:
        if args.smoke_test:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": "PPU CI",
                    "text": "### PPU CI webhook connectivity test",
                },
            }
        else:
            if args.nightly is not None:
                from ppu_notify_nightly import collect

                report = collect(args.nightly)
            else:
                report = json_file(args.report)
            weekday = WEEKDAYS[datetime.now(SHANGHAI).weekday()]
            payload = make_payload(report, os.environ.get(ROSTER, ""), weekday)
    except (OSError, KeyError, TypeError, ValueError, RecursionError):
        warn("Could not assemble the public CI notification.")
        return 0
    delivered = deliver(url, os.environ.get(SECRET, ""), payload)
    if args.smoke_test and delivered:
        print("Markdown webhook accepted the connectivity test.")
    return int(args.smoke_test and not delivered)


if __name__ == "__main__":
    # Keep the module's identity stable when the Nightly collector imports it.
    sys.modules.setdefault("ppu_notify_summary", sys.modules[__name__])
    raise SystemExit(main())
