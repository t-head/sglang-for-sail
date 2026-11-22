# SPDX-License-Identifier: Apache-2.0
"""Get the current Recipe job's Actions queue duration."""

import datetime
import json
import math
import os
from urllib.request import Request, urlopen


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    run = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    api = os.environ["GITHUB_API_URL"]
    model = os.environ["MODEL_KEY"]
    tier = os.environ["TIER"]
    title = {"fast_sanity": "Fast Sanity", "model_sanity": "Model Sanity"}[tier]
    wanted = f"{title} ({model})"
    request = Request(
        f"{api}/repos/{repo}/actions/runs/{run}/attempts/{attempt}/jobs?per_page=100",
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=10) as response:
        jobs = json.load(response)["jobs"]
    (record,) = [
        job
        for job in jobs
        if job["name"].split(" / ")[-1] == wanted and job["status"] == "in_progress"
    ]
    created, started = (
        datetime.datetime.fromisoformat(record[key].replace("Z", "+00:00"))
        for key in ("created_at", "started_at")
    )
    seconds = (started - created).total_seconds()
    if seconds < 0:
        raise ValueError("job start precedes creation")
    print(math.ceil(seconds))


if __name__ == "__main__":
    main()
