# SECOND LIGHT PRECHECK

schema `hawking.second_light.precheck.v1`  sha256 `691822a4db509f2b`  generated 2026-07-19T00:32:27Z

authoritative main: `7f237ed36a64` on `main`

## Full-run status

**full_run_status = NOT_STARTED**

> MEASURED: Second Light lease live=False, heartbeat_fresh=False, advancing=False, completed_rows=0/183. full_run_status=NOT_STARTED derived from the live controller state (fcntl flock liveness), not from any committed JSON or historical PID. Other hawking heavy owners: 0 (MoP is a separate project and is excluded).

## Mandated questions (Section 1)

| question | answer |
| --- | --- |
| Is a controller currently alive? | False |
| Is it processing the complete intended program? | False |
| What is its PID? | None |
| What lease does it hold? | None |
| What queue is advancing? | False |
| Last checkpoint time? | None |
| Percentage of complete program done? | 0.0% |

## Source receipt

- present: True  tensors: 543  shards: 7
- manifest_sha256: `d0152de427fd5e33`
- tokenizer present: True  chat_template: True

## Resources

- Mac15,14  Apple M3 Ultra  28 cores  96.0 GiB RAM
- disk free 560.2 GiB / 926.4 GiB
- thermal: Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded

## Prior ignition claims (corrected)

- `0504b0f7` claimed *one Gravity run ignited* -> UNTRUSTED per Section 0; reclassified as FIRST-LIGHT CALIBRATION; no live process, no advancing queue, no fresh heartbeat
- `80b1f1c2` claimed *120B source FAIL CLOSED (absent) -> run NOT launched* -> source-absent condition since RESOLVED; source now present+verified

## Launchd jobs

- `com.hawking.second_light` alive=False pid=None last_exit=None
- `com.hawking.doctorv5.telegram` alive=False pid=None last_exit=19200
- `com.hawking.doctorv5ultra.post120b` alive=False pid=None last_exit=512
- `com.hawking.doctorv5ultra.autoresume` alive=False pid=None last_exit=None
- `com.hawking.frontier` alive=False pid=None last_exit=None
