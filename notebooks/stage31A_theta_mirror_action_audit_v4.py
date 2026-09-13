# %% [markdown]
# Stage 31A v4 — Theta mirror-action audit on small knots
#
# Revision question:
#   The earlier mirror robustness stage treated the stored theta view as fixed
#   under mirroring because of a variable-inversion symmetry. That does not by
#   itself establish the action of taking the actual mirror of a knot.
#
# Purpose:
#   1. Query the independent Sage implementation of Theta by Bar-Natan / van der
#      Veen for several small knots and diagrammatic mirrors.
#   2. Check empirically whether theta(mirror(K)) = -theta(K) on those examples.
#   3. Audit the local archived theta table and record its feature convention.
#   4. Write a machine-readable gate JSON used by Stage 31B.
#
# IMPORTANT:
#   This stage provides an empirical encoding check on explicitly tested knots.
#   It does NOT turn the conjectural general mirror formula into a theorem.
#
# v2 changes:
#   - Uses the CURRENT SageCell /kernel + WebSocket API, not the obsolete
#     /service endpoint.
#   - Downloads Theta.sage and knots.sage from the authors' site from COLAB,
#     then bundles the source into the SageCell request because SageCell kernels
#     no longer have unrestricted outbound Internet access.
#   - Retries transient SageCell failures.
#   - Reuses a manually supplied sage_stdout.txt if it already contains AUDIT
#     lines, instead of overwriting it.
#
# Fresh-session usage in Colab:
#   %run /content/drive/MyDrive/consensus_hardness_refactored/notebooks/\
#       stage31A_theta_mirror_action_audit_v4.py
#
# Environment overrides:
#   KNOT_DATA_DIR
#   KNOT_OUTPUT_DIR
#   STAGE31A_SMALL_KNOTS="3_1,5_1,5_2,6_1,6_2,7_1,7_2,8_17"
#   STAGE31A_SAGECELL_BASE="https://sagecell.sagemath.org"
#   STAGE31A_TIMEOUT="240"
#   STAGE31A_RETRIES="3"

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Fresh-session paths
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_DIR = Path(
    "/content/drive/MyDrive/consensus_hardness_refactored"
)
DEFAULT_DATA_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants"
)
DEFAULT_ROOT = (
    DEFAULT_DATA_DIR / "processed_consensus_hardness" / "corrected_run_20260819"
)

PROJECT_DIR = Path(os.environ.get("KNOT_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
DATA_DIR = Path(os.environ.get("KNOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
ROOT = Path(os.environ.get("KNOT_OUTPUT_DIR", str(DEFAULT_ROOT)))
OUT = ROOT / "31A_theta_mirror_action_audit"

SMALL_KNOTS = tuple(
    x.strip()
    for x in os.environ.get(
        "STAGE31A_SMALL_KNOTS",
        "3_1,5_1,5_2,6_1,6_2,7_1,7_2,8_17",
    ).split(",")
    if x.strip()
)
SAGECELL_BASE = os.environ.get(
    "STAGE31A_SAGECELL_BASE", "https://sagecell.sagemath.org"
).rstrip("/")
TIMEOUT = int(os.environ.get("STAGE31A_TIMEOUT", "240"))
N_RETRIES = int(os.environ.get("STAGE31A_RETRIES", "3"))
SAGE_ENV = Path(os.environ.get("STAGE31A_SAGE_ENV", "/content/stage31a_sage_env"))
MAMBA_ROOT_PREFIX = Path(
    os.environ.get("STAGE31A_MAMBA_ROOT", "/content/stage31a_micromamba_root")
)

THETA_URL = "https://www.rolandvdv.nl/Theta/Theta.sage"
KNOTS_URL = "https://www.rolandvdv.nl/Theta/knots.sage"


def maybe_mount_drive() -> None:
    """Mount Google Drive when running in a fresh Colab runtime."""
    if "/content" not in str(Path.cwd()) and not Path("/content").exists():
        return
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore

        print("Google Drive is not mounted; requesting mount...")
        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover
        print("WARNING: could not mount Google Drive automatically:", exc)


maybe_mount_drive()

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory not found: {DATA_DIR}\n"
        "Mount Drive or set KNOT_DATA_DIR before running this stage."
    )
if not ROOT.exists():
    raise FileNotFoundError(
        f"Frozen run root not found: {ROOT}\n"
        "Mount Drive or set KNOT_OUTPUT_DIR before running this stage."
    )
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Inspect the archived theta table
# ---------------------------------------------------------------------------
THETA_PATH = DATA_DIR / "theta_upto15.csv"
if not THETA_PATH.exists():
    raise FileNotFoundError(THETA_PATH)

header = pd.read_csv(THETA_PATH, nrows=0)
theta_cols = [c for c in header.columns if str(c).startswith("T")]
if len(theta_cols) != 841:
    print(
        f"WARNING: expected 841 stored theta coordinates; found {len(theta_cols)}."
    )

pd.DataFrame({"theta_feature": theta_cols}).to_csv(
    OUT / "theta_feature_columns.csv", index=False
)

archive_audit = {
    "theta_file": str(THETA_PATH),
    "n_columns_total": int(len(header.columns)),
    "n_theta_feature_columns": int(len(theta_cols)),
    "first_20_theta_features": theta_cols[:20],
    "last_20_theta_features": theta_cols[-20:],
}
with open(OUT / "theta_archive_encoding_audit.json", "w") as fh:
    json.dump(archive_audit, fh, indent=2)

print("Archived theta coordinates:", len(theta_cols))
print("First theta columns:", theta_cols[:8])

# ---------------------------------------------------------------------------
# 2. Audit code
# ---------------------------------------------------------------------------
# Crossing triples are encoded as:
#   (crossing sign, incoming over edge, incoming under edge).
#
# Switching every crossing gives a diagram of the mirror. In this encoding we
# reverse the crossing sign and exchange over/under incoming edge roles while
# retaining the planar rotation data. The involution check below verifies that
# applying this diagram operation twice returns the original encoded diagram.
#
# The independent object being tested is then Theta itself: does evaluating the
# authors' Sage implementation on this mirror diagram give -theta(K)?

sage_knots_literal = repr(list(SMALL_KNOTS))

AUDIT_CODE = textwrap.dedent(
    f"""
    def mirror_diagram_data(K):
        X = K[0]
        rotations = list(K[1])
        Xm = matrix(ZZ, [[-ZZ(row[0]), ZZ(row[2]), ZZ(row[1])] for row in X.rows()])
        return [Xm, rotations]

    labels = {sage_knots_literal}
    for label in labels:
        K = Knot[label]
        Km = mirror_diagram_data(K)
        Kmm = mirror_diagram_data(Km)
        involution_ok = bool(Kmm[0] == K[0] and list(Kmm[1]) == list(K[1]))

        th = expand(Theta(K))
        thm = expand(Theta(Km))

        neg_ok = bool(expand(thm + th) == 0)
        same_ok = bool(expand(thm - th) == 0)

        print("AUDIT|" + label + "|" + str(involution_ok) + "|" + str(neg_ok) + "|" + str(same_ok))
        print("THETA|" + label + "|K|" + str(th))
        print("THETA|" + label + "|MIRROR|" + str(thm))
    """
)

LOCAL_DRIVER_CODE = textwrap.dedent(
    f"""
    load("{THETA_URL}")
    load("{KNOTS_URL}")
    """
) + "\n" + AUDIT_CODE

LOCAL_DRIVER = OUT / "theta_small_knot_mirror_driver.sage"
LOCAL_DRIVER.write_text(LOCAL_DRIVER_CODE)

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def stdout_has_audits(text: str) -> bool:
    return bool(text and "AUDIT|" in text)


def run_local_sage(code_path: Path) -> tuple[bool, str, str, str]:
    sage = shutil.which("sage")
    if sage is None:
        return False, "", "", "local_sage_unavailable"
    try:
        proc = subprocess.run(
            [sage, str(code_path)],
            text=True,
            capture_output=True,
            timeout=TIMEOUT,
        )
        return proc.returncode == 0, proc.stdout, proc.stderr, "local_sage"
    except Exception as exc:
        return False, "", repr(exc), "local_sage_failed"


def ensure_websocket_client() -> None:
    try:
        import websocket  # noqa: F401
        return
    except Exception:
        print("Installing websocket-client for SageCell kernel API...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "websocket-client"]
        )


def fetch_source(url: str, destination: Path) -> str:
    import requests

    last_exc: Exception | None = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Stage31A Theta audit; research reproducibility)"
    }
    for attempt in range(1, N_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            text = r.text
            if len(text.strip()) < 50:
                raise RuntimeError(f"Downloaded source from {url} is unexpectedly short.")
            destination.write_text(text)
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < N_RETRIES:
                wait = 3 * attempt
                print(
                    f"Source download attempt {attempt}/{N_RETRIES} failed for {url}: "
                    f"{exc!r}; retrying in {wait}s..."
                )
                time.sleep(wait)
    raise RuntimeError(f"Could not download {url}: {last_exc!r}")


def make_execute_request(code: str) -> str:
    session = str(uuid4())
    return json.dumps(
        {
            "channel": "shell",
            "header": {
                "msg_type": "execute_request",
                "msg_id": str(uuid4()),
                "username": "",
                "session": session,
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "user_expressions": {
                    "_sagecell_files": "sys._sage_.new_files()",
                },
                "allow_stdin": False,
            },
        }
    )


def run_sagecell_kernel(code: str) -> tuple[bool, str, str, str]:
    """
    Execute Sage through the current SageCell /kernel + websocket protocol.

    We intentionally do NOT use /service. The source files are bundled into
    `code`, so the SageCell kernel itself does not need outbound Internet.
    """
    import requests

    ensure_websocket_client()
    import websocket

    last_error = ""

    for attempt in range(1, N_RETRIES + 1):
        ws = None
        try:
            print(f"SageCell kernel attempt {attempt}/{N_RETRIES}...")

            response = requests.post(
                SAGECELL_BASE + "/kernel",
                data={"accepted_tos": "true"},
                headers={"Accept": "application/json"},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            info = response.json()

            if "id" not in info or "ws_url" not in info:
                raise RuntimeError(f"Unexpected /kernel response: {info!r}")

            ws_base = str(info["ws_url"])
            if ws_base.startswith("ws://") and SAGECELL_BASE.startswith("https://"):
                ws_base = "wss://" + ws_base[len("ws://") :]
            if not ws_base.endswith("/"):
                ws_base += "/"

            kernel_url = f"{ws_base}kernel/{info['id']}/channels"

            websocket.setdefaulttimeout(TIMEOUT)
            ws = websocket.create_connection(
                kernel_url,
                timeout=TIMEOUT,
                header=[f"Jupyter-Kernel-ID: {info['id']}"],
                origin=SAGECELL_BASE,
            )

            ws.send(make_execute_request(code))

            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            got_execute_reply = False
            got_idle_status = False
            execute_ok = True

            while not (got_execute_reply and got_idle_status):
                msg = json.loads(ws.recv())
                channel = msg.get("channel")
                header = msg.get("header", {})
                content = msg.get("content", {})
                msg_type = header.get("msg_type")

                if channel == "iopub":
                    if msg_type == "stream":
                        name = content.get("name", "stdout")
                        text = str(content.get("text", ""))
                        if name == "stderr":
                            stderr_parts.append(text)
                        else:
                            stdout_parts.append(text)
                    elif msg_type == "error":
                        execute_ok = False
                        tb = content.get("traceback", [])
                        stderr_parts.append("\n".join(map(str, tb)))
                    elif (
                        msg_type == "status"
                        and content.get("execution_state") == "idle"
                    ):
                        got_idle_status = True

                elif channel == "shell" and msg_type == "execute_reply":
                    got_execute_reply = True
                    if content.get("status") == "error":
                        execute_ok = False
                        tb = content.get("traceback", [])
                        stderr_parts.append("\n".join(map(str, tb)))

            stdout = "".join(stdout_parts)
            stderr = "\n".join(x for x in stderr_parts if x)

            if execute_ok and stdout_has_audits(stdout):
                return True, stdout, stderr, "sagecell_kernel"

            last_error = (
                "SageCell execution completed but did not produce valid AUDIT output.\n"
                + stderr[:4000]
            )

        except Exception as exc:
            last_error = repr(exc)

        finally:
            try:
                if ws is not None:
                    ws.close()
            except Exception:
                pass

        if attempt < N_RETRIES:
            wait = 5 * attempt
            print(
                f"SageCell attempt failed: {last_error[:700]}\n"
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    return False, "", last_error, "sagecell_kernel_failed"


# ---------------------------------------------------------------------------
# 4. Execute
# ---------------------------------------------------------------------------
# Fetch the independent implementation from Colab/Python first. We then load
# local snapshots inside Sage, avoiding Sage-side HTTP access.
print("Fetching independent Theta Sage sources...")
theta_source_local = fetch_source(
    THETA_URL, OUT / "Theta_source_snapshot.sage"
)
knots_source_local = fetch_source(
    KNOTS_URL, OUT / "knots_source_snapshot.sage"
)

LOCAL_DRIVER_CODE = (
    f'load("{(OUT / "Theta_source_snapshot.sage").as_posix()}")\n'
    f'load("{(OUT / "knots_source_snapshot.sage").as_posix()}")\n'
    + AUDIT_CODE
)
LOCAL_DRIVER.write_text(LOCAL_DRIVER_CODE)

STDOUT_PATH = OUT / "sage_stdout.txt"
STDERR_PATH = OUT / "sage_stderr.txt"

# A manually generated stdout is a valid explicit fallback. Do not overwrite it.
existing_stdout = ""
if STDOUT_PATH.exists():
    try:
        existing_stdout = STDOUT_PATH.read_text()
    except Exception:
        existing_stdout = ""

if stdout_has_audits(existing_stdout):
    print("Found existing sage_stdout.txt containing AUDIT output; reusing it.")
    success = True
    stdout = existing_stdout
    stderr = STDERR_PATH.read_text() if STDERR_PATH.exists() else ""
    backend = "existing_manual_stdout"
else:
    success, stdout, stderr, backend = run_local_sage(LOCAL_DRIVER)

    # Prefer a truly local Sage execution. Colab's apt repositories do not
    # consistently provide `sagemath`, so use an isolated conda-forge Sage
    # environment created with micromamba.
    if not success and os.environ.get("STAGE31A_AUTO_INSTALL_SAGE", "1") == "1":
        print("Local Sage unavailable. Bootstrapping SageMath from conda-forge...")
        install_log = OUT / "sagemath_micromamba_install.log"

        try:
            micromamba = Path("/content/stage31a_micromamba/bin/micromamba")

            if not micromamba.exists():
                micromamba.parent.parent.mkdir(parents=True, exist_ok=True)
                archive = Path("/content/stage31a_micromamba.tar.bz2")

                import requests
                url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
                print("Downloading micromamba...")
                rr = requests.get(url, timeout=TIMEOUT)
                rr.raise_for_status()
                archive.write_bytes(rr.content)

                subprocess.run(
                    [
                        "tar", "-xjf", str(archive),
                        "-C", str(micromamba.parent.parent),
                        "bin/micromamba",
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=TIMEOUT,
                )

            if not micromamba.exists():
                raise RuntimeError("micromamba executable was not extracted")

            env_sage = SAGE_ENV / "bin" / "sage"

            if not env_sage.exists():
                print(
                    "Creating isolated SageMath environment. "
                    "This is a large one-time download and may take several minutes..."
                )

                env = os.environ.copy()
                env["MAMBA_ROOT_PREFIX"] = str(MAMBA_ROOT_PREFIX)

                cmd = [
                    str(micromamba),
                    "create",
                    "-y",
                    "-p", str(SAGE_ENV),
                    "-c", "conda-forge",
                    "--strict-channel-priority",
                    "sage=10.9",
                ]

                proc = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=int(os.environ.get("STAGE31A_INSTALL_TIMEOUT", "3600")),
                )

                install_log.write_text(
                    "$ " + " ".join(cmd) + "\n\nSTDOUT:\n"
                    + proc.stdout + "\n\nSTDERR:\n" + proc.stderr
                )

                if proc.returncode != 0:
                    raise RuntimeError(
                        "micromamba Sage installation failed with exit code "
                        f"{proc.returncode}. See {install_log}"
                    )
            else:
                print("Reusing existing SageMath environment:", SAGE_ENV)

            if not env_sage.exists():
                raise RuntimeError(
                    f"Sage executable not found after installation: {env_sage}"
                )

            os.environ["PATH"] = (
                str(SAGE_ENV / "bin")
                + os.pathsep
                + os.environ.get("PATH", "")
            )

            success, stdout, stderr, backend = run_local_sage(LOCAL_DRIVER)
            if success:
                backend = "local_sage_condaforge"
            else:
                raise RuntimeError(
                    "Sage installed successfully but the Theta driver failed. "
                    f"stderr: {stderr[:1500]}"
                )

        except Exception as exc:
            try:
                previous = install_log.read_text() if install_log.exists() else ""
                install_log.write_text(
                    previous + "\n\nINSTALL/RUN ERROR:\n" + repr(exc)
                )
            except Exception:
                pass
            print("Local conda-forge Sage bootstrap failed:", repr(exc))

    if not success:
        print("Local Sage unavailable/failed; using current SageCell kernel API...")

        # Download the independent implementation OUTSIDE SageCell.
        theta_source = fetch_source(
            THETA_URL, OUT / "Theta_source_snapshot.sage"
        )
        knots_source = fetch_source(
            KNOTS_URL, OUT / "knots_source_snapshot.sage"
        )

        # SageCell user kernels cannot be assumed to have outbound Internet.
        # Bundle both source files into the request.
        bundled_code = (
            "# ---- BEGIN Theta.sage snapshot ----\n"
            + theta_source
            + "\n# ---- END Theta.sage snapshot ----\n\n"
            + "# ---- BEGIN knots.sage snapshot ----\n"
            + knots_source
            + "\n# ---- END knots.sage snapshot ----\n\n"
            + "# ---- BEGIN mirror audit ----\n"
            + AUDIT_CODE
            + "\n# ---- END mirror audit ----\n"
        )

        (OUT / "theta_small_knot_mirror_bundled_driver.sage").write_text(
            bundled_code
        )

        success, stdout, stderr, backend = run_sagecell_kernel(bundled_code)

    # Only write generated output after an actual execution attempt.
    STDOUT_PATH.write_text(stdout)
    STDERR_PATH.write_text(stderr)

if not success:
    gate = {
        "status": "UNVERIFIED",
        "reason": "Could not execute independent Sage calculation",
        "backend": backend,
        "small_knots_requested": list(SMALL_KNOTS),
        "theta_mirror_action_for_stage31B": None,
        "general_theorem_claim": False,
    }
    with open(OUT / "theta_mirror_action_gate.json", "w") as fh:
        json.dump(gate, fh, indent=2)

    raise RuntimeError(
        "Independent Sage calculation could not be executed.\n"
        f"Backend: {backend}\n"
        f"Details: {stderr[:2500]}\n\n"
        "The stage saved both the local and bundled Sage drivers in:\n"
        f"  {OUT}\n\n"
        "Do NOT run Stage 31B yet. If SageCell is temporarily unavailable, "
        "rerun this same stage later; it will preserve any valid manual "
        "sage_stdout.txt that you place in the output directory."
    )

# ---------------------------------------------------------------------------
# 5. Parse and summarize independent calculation
# ---------------------------------------------------------------------------
audit_rows: list[dict[str, Any]] = []
theta_strings: dict[tuple[str, str], str] = {}

for raw_line in stdout.splitlines():
    line = raw_line.strip()
    if line.startswith("AUDIT|"):
        parts = line.split("|")
        if len(parts) != 5:
            continue
        _, label, involution_ok, neg_ok, same_ok = parts
        audit_rows.append(
            {
                "knot": label,
                "mirror_operator_involution_ok": involution_ok == "True",
                "theta_mirror_equals_negative_theta": neg_ok == "True",
                "theta_mirror_equals_theta": same_ok == "True",
            }
        )
    elif line.startswith("THETA|"):
        parts = line.split("|", 3)
        if len(parts) == 4:
            _, label, role, poly = parts
            theta_strings[(label, role)] = poly

# Defensive normalization for quoted stream output.
if not audit_rows and "AUDIT|" in stdout:
    normalized = stdout.replace("\\n", "\n")
    for line in normalized.splitlines():
        line = line.strip()
        if line.startswith("AUDIT|"):
            parts = line.split("|")
            if len(parts) >= 5:
                audit_rows.append(
                    {
                        "knot": parts[1],
                        "mirror_operator_involution_ok": parts[2] == "True",
                        "theta_mirror_equals_negative_theta": parts[3] == "True",
                        "theta_mirror_equals_theta": parts[4] == "True",
                    }
                )

if not audit_rows:
    raise RuntimeError(
        "Sage returned but no AUDIT lines were parsed. "
        f"Inspect {STDOUT_PATH} and {STDERR_PATH}."
    )

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT / "theta_small_knot_mirror_audit.csv", index=False)

all_involution = bool(audit_df["mirror_operator_involution_ok"].all())
all_negative = bool(audit_df["theta_mirror_equals_negative_theta"].all())
all_same = bool(audit_df["theta_mirror_equals_theta"].all())

poly_rows = [
    {"knot": label, "role": role, "theta_sage": poly}
    for (label, role), poly in theta_strings.items()
]
pd.DataFrame(poly_rows).to_csv(
    OUT / "theta_small_knot_polynomials.csv", index=False
)

if all_involution and all_negative:
    action = "coefficient_sign_flip"
    status = "EMPIRICALLY_VERIFIED_ON_TESTED_SMALL_KNOTS"
else:
    action = None
    status = "FAILED_OR_AMBIGUOUS"

gate = {
    "status": status,
    "backend": backend,
    "tested_knots": audit_df["knot"].tolist(),
    "n_tested": int(len(audit_df)),
    "mirror_diagram_operator_is_involution_for_all": all_involution,
    "theta_mirror_equals_negative_theta_for_all": all_negative,
    "theta_mirror_equals_theta_for_all": all_same,
    "theta_mirror_action_for_stage31B": action,
    "general_theorem_claim": False,
    "scope_note": (
        "This is an empirical check on the listed knots using the independent "
        "Sage implementation. It does not promote the general mirror formula "
        "from conjectural to proved."
    ),
    "source_theta_sage": THETA_URL,
    "source_knots_sage": KNOTS_URL,
    "sagecell_api": SAGECELL_BASE + "/kernel",
}
with open(OUT / "theta_mirror_action_gate.json", "w") as fh:
    json.dump(gate, fh, indent=2)

print("\nIndependent small-knot mirror audit:")
print(audit_df.to_string(index=False))
print("\nGate status:", status)
print("Downstream theta action:", action)
print("Backend:", backend)
print("\nSaved Stage 31A v4 to:", OUT)

if action is None:
    raise RuntimeError(
        "Stage 31A v4 did not support a unique theta mirror action on all tested "
        "knots. Do NOT run Stage 31B until this is resolved."
    )
