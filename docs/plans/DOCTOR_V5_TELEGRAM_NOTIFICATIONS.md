# Doctor V5 Telegram rung notifications

The notifier sends one idempotent message after all four candidate branches
complete for each model at 4, 3, 2, or 1 bpw. It also sends deduplicated
blocked-execution or terminal queue alerts. Telegram delivery is operational
telemetry only and can never become Doctor evidence or promotion authority.

The token and private chat ID are stored in macOS Keychain under the `hawking`
account. They never appear in this repository, the launchd plist, notifier
launch arguments, notifier state, or logs.

## One-time setup

1. In Telegram, message `@BotFather`, send `/newbot`, and follow its prompts.
2. Run the hidden local prompt:

   ```sh
   python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py configure-token
   ```

3. Open the new bot and send it any message, such as `start`.
4. Finish setup:

   ```sh
   python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py discover-chat
   python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py prime
   python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py send-test
   python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py install
   ```

`prime` records already-completed rungs without sending historical spam.
Launchd then checks every five minutes. Restarting the Mac, queue, or notifier
cannot resend a rung because delivery state is keyed by the rung event ID
`rung/{model}/{rate}bpw`. The completed result-root hash is retained as event
metadata, not as the deduplication key.

## Status

```sh
python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py status
launchctl print gui/$(id -u)/com.hawking.doctorv5.telegram
```

Each rung message gives the GOOD/BAD decision, whether model or speed
optimization remains, the best or density-leading branch, its actual/target
physical bpw and quality deltas, and a compact Pareto pruning signal across all
four candidates. It also includes the next-block and overall ETA, wall time,
attempts, weighted progress, memory pressure, swap, disk, and thermals.
