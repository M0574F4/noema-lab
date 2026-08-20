# Noema launch-video recording steps

Recorded walkthrough: [Learned QPSK I/Q calibration receiver — complete Noema workflow](https://www.youtube.com/watch?v=bKNXS_vHLHc).
This runbook remains the versioned record of the demonstrated workflow.

Before recording, use the named shortcuts shown beside the setup commands. During recording, type
`,,` and press `Enter` for each numbered command. The visible number is the position of that command
in the `,,` shortcut sequence.

## Before recording

1. Push the launch-ready commit to the private GitHub repository.

2. Authenticate to the private repository off camera:

   ```bash
   gh auth status  # Setup shortcut P01 — type :lv-p01
   git config --global --replace-all credential.https://github.com.helper ''  # P02 — type :lv-p02
   git config --global --add credential.https://github.com.helper '!/snap/bin/gh auth git-credential'  # P03 — type :lv-p03
   git ls-remote https://github.com/M0574F4/noema-lab.git HEAD  # P04 — type :lv-p04
   ```

3. Confirm `git ls-remote` prints a commit hash without asking for credentials.

4. Create an empty recording directory:

   ```bash
   mkdir -p ~/Desktop/noema-launch-recording  # Setup shortcut P05 — type :lv-p05
   ```

5. Confirm `~/Desktop/noema-launch-recording/noema-lab` does not exist.

6. Open a terminal. This is **Terminal 1**. It will run the website server.

7. In **Terminal 1**, build the page:

   ```bash
   cd ~/Desktop/noema-lab  # Website shortcut W01 — type :lv-w01
   uv run --extra docs sphinx-build -W -b html docs .noema/launch-video/site  # W02 — type :lv-w02
   ```

8. In **Terminal 1**, start the website server:

   ```bash
   python3.12 -m http.server 8000 --directory ~/Desktop/noema-lab/.noema/launch-video/site  # W03 — type :lv-w03
   ```

9. Leave **Terminal 1** running. Never type another command in this tab unless you first stop the
   server with `Ctrl+C`.

10. Press `Ctrl+Shift+T` to open a second terminal tab. This is **Terminal 2**. It stays free for
    control commands.

11. In **Terminal 2**, open the page:

    ```bash
    xdg-open http://127.0.0.1:8000/break_the_comparison.html  # Website shortcut W04 — type :lv-w04
    ```

12. Confirm the Break the Comparison page appears in the browser.

13. Press `Ctrl+Shift+T` to open a third terminal tab. This is **Terminal 3**. It will install and run
    Noema.

14. Press `Ctrl+Shift+T` again to open a fourth terminal tab. This is **Terminal 4**. It is for
    external training commands.

15. Keep the tabs in numerical order: **Terminal 1**, **Terminal 2**, **Terminal 3**, **Terminal 4**.

16. Open **OBS > Settings > Output** and set:

    - **Output Mode:** `Simple`
    - **Recording Path:** `~/Videos`
    - **Recording Quality:** `High Quality, Medium File Size`
    - **Recording Format:** `Matroska Video (.mkv)`
    - **Video Encoder:** `Software (x264)`

17. Open **OBS > Settings > Video** and set:

    - **Base Canvas:** `1920x1080`
    - **Output Resolution:** `1920x1080`
    - **FPS:** `30`

18. Click **Apply**, then **OK**.

19. Disable notifications and close all unrelated windows.

## Record

1. Start recording in OBS.

2. In **Break the Comparison**:

   - click **Cherry-pick a seed**;
   - show the failed verdict;
   - click **Restore declared comparison**.

3. In **Terminal 3**, type `,,` and press `Enter` for each numbered command:

   ```bash
   git --version  # Shortcut 01 (:lv-s01)
   uv --version  # Shortcut 02 (:lv-s02)
   python3.12 --version  # Shortcut 03 (:lv-s03)
   cd ~/Desktop/noema-launch-recording  # Shortcut 04 (:lv-s04)
   git clone https://github.com/M0574F4/noema-lab.git  # Shortcut 05 (:lv-s05)
   cd noema-lab  # Shortcut 06 (:lv-s06)
   git rev-parse --short HEAD  # Shortcut 07 (:lv-s07)
   python3.12 tools/check_launch_video_demo.py --venv .venv  # Shortcut 08 (:lv-s08)
   uv venv .venv --python 3.12  # Shortcut 09 (:lv-s09)
   source .venv/bin/activate  # Shortcut 10 (:lv-s10)
   uv pip install -e . -r demo_trainings/neural_receiver_supervised_qpsk/requirements.txt  # Shortcut 11 (:lv-s11)
   noema --version  # Shortcut 12 (:lv-s12)
   noema ui serve --port 8766  # Shortcut 13 (:lv-s13)
   ```

4. When `http://127.0.0.1:8766` is ready, leave **Terminal 3** running. Never type another command in
   this tab unless you first stop Noema with `Ctrl+C`.

5. Cut dependency-installation waiting time if necessary.

6. Switch to the free **Terminal 2** and continue with `,,`:

   ```bash
   cd ~/Desktop/noema-launch-recording/noema-lab  # Shortcut 14 (:lv-n01)
   source .venv/bin/activate  # Shortcut 15 (:lv-n02)
   export NOEMA_VIDEO_ROOT="$PWD"  # Shortcut 16 (:lv-n03)
   export NOEMA_VIDEO_BUNDLE="$PWD/.noema/training_exports/qpsk_iq_calibration_video"  # Shortcut 17 (:lv-n04)
   xdg-open http://127.0.0.1:8766  # Shortcut 18 (:lv-n05)
   ```

7. In the browser, click:

   1. **Browse template recipes**
   2. **Physical layer & resource optimization**
   3. **Neural receiver demapping**
   4. **QPSK receiver calibration under I/Q imbalance**
   5. **Replace current pipeline**

8. Open **Workbench**.

9. Under **Operation Training Capabilities**, set **Demodulator** to **Train/replace**.

10. Under **Dataset definition > Captured signals**, select:

   - input: `receiver_frontend.rx_symbols`
   - target: `tx_bit_boundary.bits`

11. Set:

    - **Total recipe records:** `48`
    - **Train %:** `66.6667`
    - **Validation %:** `16.6667`
    - **Bundle directory:** `.noema/training_exports/qpsk_iq_calibration_video`
    - **Framework:** `PyTorch`
    - **Overwrite generated files:** off

12. Click **Export training bundle** and wait for the success message.

13. In **Terminal 2**, continue with `,,` to attach the checked-in trainer:

    ```bash
    python demo_trainings/prepare_example.py neural-receiver "$NOEMA_VIDEO_BUNDLE" --project-root "$NOEMA_VIDEO_ROOT"  # Shortcut 19 (:lv-n06)
    ```

14. Still in **Terminal 2**, manually run this check to confirm that the trainer files now exist:

    ```bash
    ls "$NOEMA_VIDEO_BUNDLE/train_demo.py" \
      "$NOEMA_VIDEO_BUNDLE/evaluate_demo.py" \
      "$NOEMA_VIDEO_BUNDLE/reference_training/build_benchmark.py"
    ```

15. Do not continue unless all three paths are printed without an error.

16. Return to **Workbench > Dataset capture** and click **Capture all datasets**.

17. Show capture starting, then cut the wait. Show train, validation, and test completed.

18. In the separate, free **Terminal 4**, continue with `,,`:

    ```bash
    cd ~/Desktop/noema-launch-recording/noema-lab  # Shortcut 20 (:lv-t01)
    source .venv/bin/activate  # Shortcut 21 (:lv-t02)
    export NOEMA_VIDEO_BUNDLE="$PWD/.noema/training_exports/qpsk_iq_calibration_video"  # Shortcut 22 (:lv-t03)
    cd "$NOEMA_VIDEO_BUNDLE"  # Shortcut 23 (:lv-t04)
    python validate_contract.py  # Shortcut 24 (:lv-t05)
    TORCH_LOGS=-onnx PYTHONWARNINGS=ignore::FutureWarning python train_demo.py  # Shortcut 25 (:lv-t06)
    python evaluate_demo.py --summary  # Shortcut 26 (:lv-t07)
    ```

19. Return to **Workbench > External model** and click **Validate returned model**.

20. Show **Validation complete** and **model interface valid**.

21. Stop using `,,`; command 26 is the final recording shortcut. The comparison is created and run
    entirely in the existing Noema UI.

22. Open **Graph**. Keep the original recipe tab as the uncompensated baseline.

23. Open **Recipe settings**, set **Recipe name** to `Uncompensated QPSK`, then close the dialog.

24. Click the recipe tab's **Copy** icon.

25. On the copied tab, open **Recipe settings**, set **Recipe name** to `Calibrated I/Q oracle`, then
    close the dialog.

26. Open the **Demodulator** block and set **Mode** to **Calibrated I/Q oracle**.

27. Click this recipe tab's **Copy** icon.

28. On the new tab, open **Recipe settings**, set **Recipe name** to `Learned I/Q receiver`, then close
    the dialog.

29. Open the **Demodulator** block and set **Mode** to **Learned model**.

30. Under **Model artifact**, select **Learned receiver · qpsk_iq_imbalance_receiver_calibration**.

31. Confirm that three recipe tabs are open and each retains the `-2`, `2`, `6`, and `10` dB SNR
    matrix.

32. Click **Run All**. Show all three tabs running, then cut the wait.

33. Do not continue until all three tabs show completed runs.

34. Open **Results**. Do not select **Past runs** or **Benchmark results**; the open recipe tabs are
    already the active comparison.

35. Open **Communication** and show **Channel BER** and **QPSK receiver decision boundaries**.

36. Hover these series in order:

    1. **Uncompensated QPSK**
    2. **Calibrated I/Q oracle**
    3. **Learned I/Q receiver**

37. Show the received-I/Q decision boundaries.

38. Return to **Break the Comparison** for the final frame.

39. Stop recording in OBS.

40. In OBS, select **File > Remux Recordings** and create the MP4.

41. Watch the complete MP4 and check that no credentials, notifications, or private windows appear.
