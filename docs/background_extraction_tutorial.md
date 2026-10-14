# Tutorial: Background Extraction with Guarded Checkpointing and Tmux

This tutorial explains how to run the multi-gigabyte Objects365 duck dataset extraction reliably in the background, pause and resume at any time without data loss, and monitor real-time progress using `tmux` or native Windows PowerShell.

---

## 1. Why Long Extractions Fail in Notebooks

Running long network-bound extraction loops (>1 hour) inside interactive Jupyter Notebook cells presents several failure modes:
* **Browser Disconnection**: Closing the browser tab or reloading the page often severs the WebSocket connection to the kernel.
* **OS Sleep Mode**: If your laptop goes to sleep or turns off the display, the notebook kernel is abruptly killed.
* **Corrupted Half-Downloads**: An interrupted cell execution may leave half-downloaded multi-hundred-megabyte `.tar.gz` files on disk, requiring manual cleanup.

To solve this, the pipeline includes a **Guarded Checkpoint Manager** that runs independently of your editor and browser.

---

## 2. Guarded Checkpointing Architecture

The extraction engine (`src/data/objects365.py`) provides four layers of resilience:

1. **Persistent Checkpoint Ledger (`data/checkpoint_val.json`)**:
   Tracks the exact state of every patch archive (`completed_patches`, `completed_images`, `in_progress_patch`). Completed patches are skipped in under 0.01 seconds upon restart.

2. **HTTP Range-Resumed Downloads**:
   If an internet drop interrupts an archive download midway (e.g., at 320 MB of 500 MB), the downloader automatically issues a `Range: bytes=335544320-` header upon restart, downloading only the remaining bytes.

3. **Graceful Signal Handling (Safe Pausing)**:
   Catches `SIGINT` (Ctrl+C) and `SIGTERM` signals. When you stop the process, it safely flushes all progress to disk before exiting cleanly.

4. **Minimal Disk Footprint**:
   Only one patch archive exists on disk at any given time (~500 MB). The archive is unlinked immediately after extracting the duck images, ensuring disk usage remains low throughout the 14 GB extraction.

---

## 3. Method A: Running with Tmux via WSL (Recommended)

`tmux` allows a command to run in an independent terminal session on your computer. You can detach, close your terminal, shut down VS Code, and reattach anytime.

### Step 1: Start a New Tmux Session
Open PowerShell or Command Prompt and run:
```bash
wsl tmux new -s duck_extract
```

### Step 2: Navigate and Run the Extractor
Inside the tmux window:
```bash
cd /mnt/d/DataNLP/duck
/mnt/d/DataNLP/duck/.venv/Scripts/python.exe scripts/extract_objects365.py --split val
```

### Step 3: Detach from the Session
* Press `Ctrl + B`, then release both keys and press `D`.
* You will return to your host terminal with:
  ```text
  [detached (from session duck_extract)]
  ```
* **You can now safely close your terminal, exit your IDE, or let your laptop continue running in the background.**

### Step 4: Reattach and Monitor Progress
Whenever you want to check the live download and extraction stream:
```bash
wsl tmux attach -t duck_extract
```

### Step 5: Stop and Resume
* **To Pause**: Inside the attached tmux session, press `Ctrl + C`. The script catches the interrupt, updates `data/checkpoint_val.json`, and exits cleanly.
* **To Resume**: Re-run the same command inside tmux:
  ```bash
  /mnt/d/DataNLP/duck/.venv/Scripts/python.exe scripts/extract_objects365.py --split val
  ```
  The checkpoint manager instantly skips all already completed patches and continues directly from the next one.

---

## 4. Method B: Native Windows PowerShell Background Daemon

If you do not want to use WSL, you can run the process natively in the Windows background using PowerShell.

### Step 1: Launch Background Process with Log Redirection
From the project root in PowerShell:
```powershell
Start-Process -NoNewWindow -FilePath ".venv\Scripts\python.exe" -ArgumentList "scripts/extract_objects365.py --split val" -RedirectStandardOutput "logs/extract.log" -RedirectStandardError "logs/extract_err.log"
```

### Step 2: Stream Live Progress
Monitor the log output in real-time:
```powershell
Get-Content logs/extract.log -Wait -Tail 25
```
*(Press `Ctrl + C` at any time to stop viewing the log; the background extraction process will keep running).*

### Step 3: Stop the Background Process
```powershell
Stop-Process -Name python
```

### Step 4: Resume
Re-run the `Start-Process` command from Step 1. The checkpoint ledger automatically resumes from the latest completed state.

---

## 5. Checking Extraction Status Dashboard

At any point, whether the process is running or paused, check your exact progress with:

```bash
python scripts/extract_objects365.py --status
```

Sample Dashboard Output:
```text
=================================================================
  DUCK EXTRACTION STATUS: Objects365 (VAL SPLIT)
=================================================================
  * Total Duck Images Target:   1,547
  * Total Duck Bounding Boxes:  12,221
  * Images Currently On Disk:   637 (41.2%)
  * Completed Patches:          19 patches
  * In-Progress Patch:          None (idle)
  * Checkpoint Last Updated:    2026-10-14T13:08:21Z
=================================================================
```
