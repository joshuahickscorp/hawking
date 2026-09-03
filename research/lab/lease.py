from __future__ import annotations
import fcntl
import json
import os
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO, Type
from lab.receipts import seal as _seal_doc
from lab.receipts import _sha256_hex as sha256_hex, _utc_now as utc_now
LEASE_SCHEMA = 'hawking.lab.controller_lease.v1'
FIXTURE_HEAVY_LEASE_SCHEMA = 'hawking.lab.fixture_heavy_lease.v1'
_PROCESS_LEASES: set[str] = set()
_PROCESS_LEASES_LOCK = threading.Lock()

class LeaseError(RuntimeError):
    pass


class FixtureLeaseError(LeaseError):
    """A fixture-only heavy lease was not clean, fresh, or exclusive."""

def _verify_sealed(value: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    recorded = value.get('seal_sha256')
    unsigned = {k: v for k, v in value.items() if k != 'seal_sha256'}
    return recorded == sha256_hex(unsigned)

class SingletonLease:

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        campaign_id: str,
        controller_epoch: str='1',
        owner: str='lab-engine',
        lease_schema: str=LEASE_SCHEMA,
        error_type: Type[BaseException] | None=None,
        seal_owner: bool=True,
    ) -> None:
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.controller_epoch = controller_epoch
        self.owner = owner
        self.lease_schema = lease_schema
        self.error_type: Type[BaseException] = error_type or LeaseError
        self.seal_owner = seal_owner
        self._handle: TextIO | None = None
        self._registry_key: str | None = None

    def _fail(self, message: str) -> None:
        raise self.error_type(message)

    @property
    def held(self) -> bool:
        return self._handle is not None

    def assert_held(self) -> None:
        if not self.held:
            self._fail('controller mutation refused: singleton lease is not held')

    def assert_path_bound(self) -> None:
        """Prove the locked descriptor is still the file named by ``path``."""
        self.assert_held()
        assert self._handle is not None
        try:
            descriptor = os.fstat(self._handle.fileno())
            named = os.lstat(os.path.abspath(os.path.expanduser(os.fspath(self.path))))
        except OSError as exc:
            raise self.error_type(f'singleton lease path changed while held: {exc}') from exc
        self._require_safe_lease_file(descriptor, label='held singleton lease descriptor')
        self._require_safe_lease_file(named, label='held singleton lease path')
        if self._identity(descriptor) != self._identity(named):
            self._fail('singleton lease path was replaced while the locked descriptor remained held')

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
        return (int(metadata.st_dev), int(metadata.st_ino), stat.S_IFMT(metadata.st_mode))

    @classmethod
    def _file_fingerprint(cls, metadata: os.stat_result) -> tuple[int, ...]:
        return (
            *cls._identity(metadata),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_uid),
            int(metadata.st_gid),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    def _require_safe_lease_file(self, metadata: os.stat_result, *, label: str) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            self._fail(f'{label} is not a regular file')
        if int(metadata.st_nlink) != 1:
            self._fail(f'{label} must have exactly one hard link')
        get_euid = getattr(os, 'geteuid', None)
        if callable(get_euid) and int(metadata.st_uid) != int(get_euid()):
            self._fail(f'{label} is not owned by the current user')

    def _verify_directory_chain(
        self,
        descriptors: Sequence[int],
        links: Sequence[tuple[str, tuple[int, int, int]]],
        *,
        label: str,
    ) -> None:
        if len(descriptors) != len(links) + 1:
            self._fail(f'{label} descriptor chain is malformed')
        for index, (component, expected) in enumerate(links):
            try:
                named = os.stat(component, dir_fd=descriptors[index], follow_symlinks=False)
                opened = os.fstat(descriptors[index + 1])
            except OSError as exc:
                raise self.error_type(f'{label} changed while acquiring lease: {component!r}: {exc}') from exc
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or self._identity(named) != expected
                or self._identity(opened) != expected
            ):
                self._fail(f'{label} identity changed while acquiring lease: {component!r}')

    def _open_parent_chain(self) -> tuple[str, str, list[int], list[tuple[str, tuple[int, int, int]]]]:
        nofollow = getattr(os, 'O_NOFOLLOW', 0)
        directory = getattr(os, 'O_DIRECTORY', 0)
        cloexec = getattr(os, 'O_CLOEXEC', 0)
        if not nofollow or not directory or (not cloexec):
            self._fail('singleton lease acquisition requires O_NOFOLLOW, O_DIRECTORY, and O_CLOEXEC')
        expanded = os.path.abspath(os.path.expanduser(os.fspath(self.path)))
        leaf = os.path.basename(expanded)
        if leaf in {'', '.', '..'}:
            self._fail('singleton lease path must name a file')
        parent = str(Path(os.path.dirname(expanded)).resolve(strict=False))
        absolute_path = os.path.join(parent, leaf)
        components = tuple((item for item in parent.split(os.sep) if item))
        flags = os.O_RDONLY | nofollow | directory | cloexec
        descriptors: list[int] = []
        links: list[tuple[str, tuple[int, int, int]]] = []
        try:
            named_root = os.stat(os.sep, follow_symlinks=False)
            root_fd = os.open(os.sep, flags)
            descriptors.append(root_fd)
            opened_root = os.fstat(root_fd)
            if self._identity(named_root) != self._identity(opened_root) or not stat.S_ISDIR(opened_root.st_mode):
                self._fail('filesystem root changed while opening singleton lease')
            for component in components:
                try:
                    named = os.stat(component, dir_fd=descriptors[-1], follow_symlinks=False)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, mode=511, dir_fd=descriptors[-1])
                    except FileExistsError:
                        pass
                    named = os.stat(component, dir_fd=descriptors[-1], follow_symlinks=False)
                if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                    self._fail(f'singleton lease parent component is not a real directory: {component!r}')
                child_fd = os.open(component, flags, dir_fd=descriptors[-1])
                try:
                    opened = os.fstat(child_fd)
                except OSError:
                    os.close(child_fd)
                    raise
                if self._identity(named) != self._identity(opened):
                    os.close(child_fd)
                    self._fail(f'singleton lease parent component changed while opening: {component!r}')
                links.append((component, self._identity(opened)))
                descriptors.append(child_fd)
            self._verify_directory_chain(descriptors, links, label='singleton lease parent')
            return (absolute_path, leaf, descriptors, links)
        except BaseException as exc:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            if isinstance(exc, OSError):
                raise self.error_type(f'cannot securely open singleton lease parent: {exc}') from exc
            if isinstance(exc, self.error_type):
                raise
            raise

    def _after_lock_before_prewrite_revalidation(self, parent_fd: int, leaf: str, lease_fd: int) -> None:
        pass

    def _open_lease_file(self, parent_fd: int, leaf: str) -> tuple[int, os.stat_result]:
        flags = os.O_RDWR | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
        for _ in range(16):
            try:
                named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    descriptor = os.open(leaf, flags | os.O_CREAT | os.O_EXCL, 438, dir_fd=parent_fd)
                except FileExistsError:
                    continue
                try:
                    opened = os.fstat(descriptor)
                    named_created = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                    self._require_safe_lease_file(opened, label='new singleton lease file')
                    if self._file_fingerprint(named_created) != self._file_fingerprint(opened):
                        self._fail('new singleton lease name changed while creating')
                    return (descriptor, opened)
                except BaseException:
                    os.close(descriptor)
                    raise
            self._require_safe_lease_file(named, label='singleton lease file')
            descriptor = os.open(leaf, flags, dir_fd=parent_fd)
            try:
                opened = os.fstat(descriptor)
                self._require_safe_lease_file(opened, label='singleton lease descriptor')
                if self._file_fingerprint(named) != self._file_fingerprint(opened):
                    self._fail('singleton lease file changed while opening')
                return (descriptor, opened)
            except BaseException:
                os.close(descriptor)
                raise
        self._fail('singleton lease path did not stabilize while opening')
        raise AssertionError('unreachable')

    def acquire(self, *, blocking: bool=False) -> 'SingletonLease':
        if blocking:
            self._fail('blocking lease acquisition is not supported')
        if self.held:
            self._fail('singleton lease already held by this handle')
        absolute_path, leaf, parent_descriptors, parent_links = self._open_parent_chain()
        key = absolute_path
        with _PROCESS_LEASES_LOCK:
            if key in _PROCESS_LEASES:
                for parent_descriptor in reversed(parent_descriptors):
                    os.close(parent_descriptor)
                self._fail(f'already-running: singleton lease held in this process: {self.path}')
            _PROCESS_LEASES.add(key)
        handle: TextIO | None = None
        descriptor: int | None = None
        try:
            parent_fd = parent_descriptors[-1]
            descriptor, descriptor_pre = self._open_lease_file(parent_fd, leaf)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise self.error_type(f'already-running: singleton lease held: {self.path}') from exc
            self._after_lock_before_prewrite_revalidation(parent_fd, leaf, descriptor)
            descriptor_post_lock = os.fstat(descriptor)
            self._require_safe_lease_file(descriptor_post_lock, label='locked singleton lease descriptor')
            try:
                named_post_lock = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise self.error_type(f'singleton lease name changed before owner-record write: {exc}') from exc
            self._require_safe_lease_file(named_post_lock, label='locked singleton lease file')
            if (
                self._file_fingerprint(descriptor_pre)
                != self._file_fingerprint(descriptor_post_lock)
                or self._file_fingerprint(named_post_lock)
                != self._file_fingerprint(descriptor_post_lock)
            ):
                self._fail('singleton lease file changed before owner-record write')
            self._verify_directory_chain(parent_descriptors, parent_links, label='singleton lease parent')
            stamp: dict[str, Any] = {
                'schema': self.lease_schema,
                'campaign_id': self.campaign_id,
                'controller_epoch': self.controller_epoch,
                'owner': self.owner,
                'pid': os.getpid(),
                'acquired_at': utc_now(),
            }
            if self.seal_owner:
                stamp = _seal_doc(stamp)
            encoded = (json.dumps(stamp, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    self._fail(f'short owner-record write to {self.path}')
                view = view[written:]
            os.fsync(descriptor)
            named_after_write = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            descriptor_after_write = os.fstat(descriptor)
            self._require_safe_lease_file(descriptor_after_write, label='written singleton lease descriptor')
            if self._file_fingerprint(named_after_write) != self._file_fingerprint(descriptor_after_write):
                self._fail('singleton lease name changed during owner-record write')
            self._verify_directory_chain(parent_descriptors, parent_links, label='singleton lease parent')
            os.fsync(parent_fd)
            handle = os.fdopen(descriptor, 'r+', encoding='utf-8')
            descriptor = None
            self._handle = handle
            self._registry_key = key
            return self
        except BaseException:
            if handle is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                handle.close()
            elif descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)
            with _PROCESS_LEASES_LOCK:
                _PROCESS_LEASES.discard(key)
            raise
        finally:
            for parent_descriptor in reversed(parent_descriptors):
                os.close(parent_descriptor)

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            if self._registry_key is not None:
                with _PROCESS_LEASES_LOCK:
                    _PROCESS_LEASES.discard(self._registry_key)
                self._registry_key = None
    close = release

    def read_owner(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def probe(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                'lock_state': 'ABSENT',
                'live_lock_held': False,
                'held_by_this_handle': self.held,
                'owner_record_ok': False,
                'owner': None,
                'owner_pid': None,
                'owner_pid_alive': False,
                'controller_epoch': None,
            }
        live_lock = self.held
        descriptor: int | None = None
        if not self.held:
            try:
                descriptor = os.open(self.path, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    live_lock = True
                else:
                    live_lock = False
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                live_lock = False
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        owner_record: dict[str, Any] | None = None
        try:
            raw = json.loads(self.path.read_bytes())
            if not isinstance(raw, dict):
                owner_record = None
            elif self.seal_owner and (not _verify_sealed(raw)):
                owner_record = None
            elif raw.get('schema') != self.lease_schema or raw.get('campaign_id') != self.campaign_id:
                owner_record = None
            else:
                owner_record = raw
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            owner_record = None
        owner_pid = owner_record.get('pid') if owner_record is not None else None
        owner_pid_alive = False
        if not isinstance(owner_pid, bool) and isinstance(owner_pid, int) and (owner_pid > 0):
            try:
                os.kill(owner_pid, 0)
                owner_pid_alive = True
            except PermissionError:
                owner_pid_alive = True
            except ProcessLookupError:
                owner_pid_alive = False
        return {
            'lock_state': 'HELD_BY_THIS_HANDLE' if self.held else 'HELD_BY_OTHER_PROCESS' if live_lock else 'UNLOCKED',
            'live_lock_held': live_lock,
            'held_by_this_handle': self.held,
            'owner_record_ok': owner_record is not None,
            'owner': owner_record.get('owner') if owner_record is not None else None,
            'owner_pid': owner_pid,
            'owner_pid_alive': owner_pid_alive,
            'controller_epoch': owner_record.get('controller_epoch') if owner_record is not None else None,
        }

    def __enter__(self) -> 'SingletonLease':
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


class FixtureHeavyLease:
    """Heartbeat-backed wrapper around the sole hardened ``SingletonLease``.

    It is deliberately incapable of authorizing production work.  Callers
    must label their resource state CLEAN, CONTENDED, or INVALID and present
    the known foreign-process set before beginning a fixture heavy operation.
    A non-empty foreign set changes the label to CONTENDED and refuses work.
    Recovery never breaks a live lock: it only records that an *unlocked*
    stale owner record was replaced by a new exclusive holder.
    """

    _LABELS = frozenset({'CLEAN', 'CONTENDED', 'INVALID'})

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        campaign_id: str,
        owner: str='fixture-window-operator',
        heartbeat_timeout_seconds: float=30.0,
        wall_clock_ns: Any=time.time_ns,
        monotonic_clock_ns: Any=time.monotonic_ns,
    ) -> None:
        if (
            isinstance(heartbeat_timeout_seconds, bool)
            or not isinstance(heartbeat_timeout_seconds, (int, float))
            or heartbeat_timeout_seconds <= 0
        ):
            raise FixtureLeaseError('heartbeat_timeout_seconds must be positive')
        self._lease = SingletonLease(
            path,
            campaign_id=campaign_id,
            controller_epoch='fixture-heavy-v1',
            owner=owner,
            lease_schema=FIXTURE_HEAVY_LEASE_SCHEMA,
            error_type=FixtureLeaseError,
            seal_owner=True,
        )
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.owner = owner
        self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        if not callable(wall_clock_ns) or not callable(monotonic_clock_ns):
            raise FixtureLeaseError('fixture lease clocks must be callable')
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_clock_ns = monotonic_clock_ns
        self._process_identity = secrets.token_hex(32)
        self.lease_id = secrets.token_hex(32)
        self._record: dict[str, Any] | None = None

    @property
    def held(self) -> bool:
        return self._lease.held

    @staticmethod
    def _foreign_rows(foreign_processes: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        if foreign_processes is None:
            return []
        if isinstance(foreign_processes, (str, bytes)):
            raise FixtureLeaseError('foreign_processes must be mappings, not text')
        rows: list[dict[str, Any]] = []
        for number, value in enumerate(foreign_processes):
            if not isinstance(value, Mapping):
                raise FixtureLeaseError(f'foreign process {number} is not a mapping')
            pid = value.get('pid')
            label = value.get('label')
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise FixtureLeaseError(f'foreign process {number} pid is invalid')
            if not isinstance(label, str) or not label or label != label.strip():
                raise FixtureLeaseError(f'foreign process {number} label is invalid')
            rows.append({'pid': pid, 'label': label})
        return rows

    def _validate_label(self, label: str) -> str:
        if label not in self._LABELS:
            raise FixtureLeaseError(f'contention label must be one of {sorted(self._LABELS)!r}')
        return label

    def _write_record(self, *, contention_label: str, foreign_processes: list[dict[str, Any]], recovered: bool) -> None:
        self._lease.assert_held()
        self._lease.assert_path_bound()
        handle = self._lease._handle
        if handle is None:
            raise FixtureLeaseError('fixture lease handle disappeared')
        now_ns = self._wall_clock_ns()
        monotonic_ns = self._monotonic_clock_ns()
        value = _seal_doc({
            'schema': FIXTURE_HEAVY_LEASE_SCHEMA,
            'campaign_id': self.campaign_id,
            'controller_epoch': 'fixture-heavy-v1',
            'owner': self.owner,
            'pid': os.getpid(),
            'process_identity': self._process_identity,
            'acquired_at': (self._record or {}).get('acquired_at', utc_now()),
            'lease_id': self.lease_id,
            'fixture_only': True,
            'production_authority': False,
            'contention_label': contention_label,
            'foreign_processes': foreign_processes,
            'heartbeat_unix_ns': now_ns,
            'heartbeat_monotonic_ns': monotonic_ns,
            'heartbeat_timeout_seconds': self.heartbeat_timeout_seconds,
            'recovered_unlocked_stale_record': recovered,
        })
        encoded = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')
        descriptor = handle.fileno()
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FixtureLeaseError('short fixture heartbeat write')
            view = view[written:]
        os.fsync(descriptor)
        self._lease.assert_path_bound()
        self._record = value

    def _classify_prior_unlocked_record(self, prior: dict[str, Any] | None) -> bool:
        """Return whether a valid prior record is stale enough to recover.

        An unlocked file is not itself proof of staleness: a fresh owner record
        can result from accidental lock loss, PID reuse, or an attacker swapping
        lock state.  Only an authenticated, expired fixture heartbeat is
        recoverable.  Any malformed or fresh record fails closed.
        """
        if prior is None:
            return False
        if (
            not _verify_sealed(prior)
            or prior.get('schema') != FIXTURE_HEAVY_LEASE_SCHEMA
            or prior.get('campaign_id') != self.campaign_id
            or prior.get('fixture_only') is not True
            or prior.get('production_authority') is not False
        ):
            raise FixtureLeaseError('unlocked fixture lease owner record is invalid; explicit quarantine is required')
        heartbeat = prior.get('heartbeat_unix_ns')
        if isinstance(heartbeat, bool) or not isinstance(heartbeat, int):
            raise FixtureLeaseError('unlocked fixture lease owner heartbeat is invalid')
        age_ns = self._wall_clock_ns() - heartbeat
        if age_ns < 0:
            raise FixtureLeaseError('unlocked fixture lease owner heartbeat is from the future')
        recorded_timeout = prior.get('heartbeat_timeout_seconds')
        if (
            isinstance(recorded_timeout, bool)
            or not isinstance(recorded_timeout, (int, float))
            or recorded_timeout <= 0
        ):
            raise FixtureLeaseError('unlocked fixture lease owner timeout is invalid')
        # The prior owner's authenticated timeout is the recovery contract;
        # a new claimant may neither shorten it nor retroactively lengthen it.
        timeout_ns = int(float(recorded_timeout) * 1_000_000_000)
        if age_ns <= timeout_ns:
            raise FixtureLeaseError('unlocked fixture lease owner record is still fresh; recovery refused')
        return True

    def acquire(
        self,
        *,
        contention_label: str='CLEAN',
        foreign_processes: Iterable[Mapping[str, Any]] | None=None,
    ) -> 'FixtureHeavyLease':
        if os.environ.get('HAWKING_PARENT_RESTREAM_AUTHORIZED') == 'YES':
            raise FixtureLeaseError('fixture-only lease cannot accompany parent-restream authorization')
        label = self._validate_label(contention_label)
        foreign = self._foreign_rows(foreign_processes)
        if foreign:
            label = 'CONTENDED'
        prior = self._lease.read_owner()
        probe = self._lease.probe()
        if probe['live_lock_held']:
            raise FixtureLeaseError('fixture heavy lease is already held; recovery may not break a live lock')
        recovered = self._classify_prior_unlocked_record(prior)
        self._lease.acquire()
        self._write_record(
            contention_label=label,
            foreign_processes=foreign,
            recovered=recovered,
        )
        if foreign:
            self.release()
            raise FixtureLeaseError('foreign-process protection refused a contended fixture lease')
        if label != 'CLEAN':
            self.release()
            raise FixtureLeaseError('fixture heavy work requires an explicit CLEAN contention label')
        return self

    def heartbeat(
        self,
        *,
        contention_label: str='CLEAN',
        foreign_processes: Iterable[Mapping[str, Any]] | None=None,
    ) -> None:
        label = self._validate_label(contention_label)
        foreign = self._foreign_rows(foreign_processes)
        if foreign:
            label = 'CONTENDED'
        self._write_record(contention_label=label, foreign_processes=foreign, recovered=False)
        if foreign or label != 'CLEAN':
            raise FixtureLeaseError('fixture heartbeat reports contention; operation must roll back')

    def assert_clean(self) -> None:
        self._lease.assert_held()
        try:
            self._lease.assert_path_bound()
        except FixtureLeaseError as exc:
            raise FixtureLeaseError(f'fixture lease record was replaced or tampered while held: {exc}') from exc
        record = self._record
        if record is None or record.get('fixture_only') is not True or record.get('production_authority') is not False:
            raise FixtureLeaseError('fixture-only lease record is malformed')
        if record.get('contention_label') != 'CLEAN' or record.get('foreign_processes') != []:
            raise FixtureLeaseError('fixture heavy operation is not CLEAN')
        on_disk = self._lease.read_owner()
        if (
            on_disk is None
            or not _verify_sealed(on_disk)
            or on_disk != record
            or on_disk.get('lease_id') != self.lease_id
            or on_disk.get('process_identity') != self._process_identity
        ):
            raise FixtureLeaseError('fixture lease record was replaced or tampered while held')
        heartbeat = record.get('heartbeat_unix_ns')
        monotonic = record.get('heartbeat_monotonic_ns')
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (heartbeat, monotonic)):
            raise FixtureLeaseError('fixture heartbeat is invalid')
        now_wall = self._wall_clock_ns()
        now_monotonic = self._monotonic_clock_ns()
        wall_age = now_wall - heartbeat
        monotonic_age = now_monotonic - monotonic
        if wall_age < 0 or monotonic_age < 0:
            raise FixtureLeaseError('fixture heartbeat clock regressed; reacquire before work')
        timeout_ns = int(self.heartbeat_timeout_seconds * 1_000_000_000)
        if max(wall_age, monotonic_age) > timeout_ns:
            raise FixtureLeaseError('fixture heartbeat is stale; recover or reacquire before work')

    def receipt(self) -> dict[str, Any]:
        self.assert_clean()
        assert self._record is not None
        return dict(self._record)

    def release(self) -> None:
        self._lease.release()
        self._record = None

    close = release

    def __enter__(self) -> 'FixtureHeavyLease':
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()
