//! Exact native I/O helpers used only by the opt-in native execution binary.
//!
//! These helpers change allocation and file-growth strategy, never framing or
//! payload bytes. They deliberately do not install a process-global allocator.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static WORKER_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn read_preallocated(path: &Path) -> io::Result<Vec<u8>> {
    let len = usize::try_from(fs::metadata(path)?.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::FileTooLarge,
            "input length does not fit in usize",
        )
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|error| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!("cannot reserve {len} input bytes: {error}"),
        )
    })?;
    bytes.resize(len, 0);
    File::open(path)?.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub fn create_preallocated(path: &Path, len: u64) -> io::Result<File> {
    // `create_new` is atomic and refuses every existing directory entry,
    // including symlinks, so workers never follow or truncate a planted path.
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.set_len(len)?;
    file.seek(SeekFrom::Start(0))?;
    Ok(file)
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

/// One same-directory, worker-owned output which is invisible at its requested
/// name until all bytes and append sections are durable.
///
/// Publication uses an exclusive hard link rather than `rename`: creating the
/// final name is atomic and cannot replace an artifact owned by another worker.
pub struct WorkerOutput {
    target: PathBuf,
    temporary: PathBuf,
    file: Option<File>,
    cleanup_temporary: bool,
}

impl WorkerOutput {
    pub fn create(target: &Path, len: u64) -> io::Result<Self> {
        let parent = target.parent().unwrap_or_else(|| Path::new("."));
        let name = target.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "output path has no file name")
        })?;
        if fs::symlink_metadata(target).is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "refusing to replace an existing output entry",
            ));
        }
        let mut selected = None;
        for _ in 0..64 {
            let sequence = WORKER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let mut temporary_name = OsString::from(".");
            temporary_name.push(name);
            temporary_name.push(format!(".worker-{}-{sequence}.tmp", std::process::id()));
            let temporary = parent.join(temporary_name);
            match create_preallocated(&temporary, len) {
                Ok(file) => {
                    selected = Some((temporary, file));
                    break;
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        let (temporary, file) = selected.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a unique worker output",
            )
        })?;
        Ok(Self {
            target: target.to_path_buf(),
            temporary,
            file: Some(file),
            cleanup_temporary: true,
        })
    }

    pub fn file_mut(&mut self) -> io::Result<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "worker output writer is closed")
        })
    }

    pub fn temporary_path(&self) -> &Path {
        &self.temporary
    }

    /// Flush and close the initial writer so path-based append helpers may open
    /// the worker-owned temporary without racing buffered bytes.
    pub fn close_writer(&mut self) -> io::Result<()> {
        if let Some(mut file) = self.file.take() {
            file.flush()?;
            file.sync_all()?;
        }
        Ok(())
    }

    /// Durably publish the complete temporary at the requested staging name.
    pub fn finalize(mut self) -> io::Result<()> {
        self.close_writer()?;
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.temporary)?
            .sync_all()?;
        let parent = self.target.parent().unwrap_or_else(|| Path::new("."));
        // Persist the exclusively-created temporary before adding the final link.
        sync_directory(parent)?;
        fs::hard_link(&self.temporary, &self.target)?;
        // Once the final name exists, preserve the temporary on any later error
        // so an operator has a recovery artifact rather than an ambiguous loss.
        self.cleanup_temporary = false;
        sync_directory(parent)?;
        fs::remove_file(&self.temporary)?;
        sync_directory(parent)?;
        Ok(())
    }
}

impl Drop for WorkerOutput {
    fn drop(&mut self) {
        if self.cleanup_temporary {
            let _ = fs::remove_file(&self.temporary);
        }
    }
}

pub fn write_preallocated(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut output = WorkerOutput::create(path, bytes.len() as u64)?;
    output.file_mut()?.write_all(bytes)?;
    output.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preallocated_read_write_is_exact() {
        let path = std::env::temp_dir().join(format!(
            "strand-native-io-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_file(&path);
        let expected = (0..65_537)
            .map(|index| (index as u8).wrapping_mul(37))
            .collect::<Vec<_>>();
        write_preallocated(&path, &expected).unwrap();
        assert_eq!(
            write_preallocated(&path, b"replacement")
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::metadata(&path).unwrap().len(), expected.len() as u64);
        assert_eq!(read_preallocated(&path).unwrap(), expected);
        let _ = fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn preallocated_output_refuses_symlink() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir();
        let target = root.join(format!("strand-native-target-{}", std::process::id()));
        let link = root.join(format!("strand-native-link-{}.partial", std::process::id()));
        let _ = fs::remove_file(&target);
        let _ = fs::remove_file(&link);
        fs::write(&target, b"owner").unwrap();
        symlink(&target, &link).unwrap();
        assert!(write_preallocated(&link, b"replacement").is_err());
        assert_eq!(fs::read(&target).unwrap(), b"owner");
        let _ = fs::remove_file(link);
        let _ = fs::remove_file(target);
    }

    #[test]
    fn worker_output_supports_durable_append_before_publication() {
        let root = std::env::temp_dir();
        let target = root.join(format!(
            "strand-native-append-{}.partial",
            std::process::id()
        ));
        let _ = fs::remove_file(&target);
        let mut output = WorkerOutput::create(&target, 4).unwrap();
        output.file_mut().unwrap().write_all(b"base").unwrap();
        let temporary = output.temporary_path().to_path_buf();
        assert!(!target.exists());
        output.close_writer().unwrap();
        OpenOptions::new()
            .append(true)
            .open(&temporary)
            .unwrap()
            .write_all(b"-append")
            .unwrap();
        output.finalize().unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"base-append");
        assert!(!temporary.exists());
        let _ = fs::remove_file(target);
    }

    #[test]
    fn dropped_worker_output_removes_unpublished_temporary() {
        let target =
            std::env::temp_dir().join(format!("strand-native-drop-{}.partial", std::process::id()));
        let _ = fs::remove_file(&target);
        let temporary = {
            let output = WorkerOutput::create(&target, 0).unwrap();
            output.temporary_path().to_path_buf()
        };
        assert!(!target.exists());
        assert!(!temporary.exists());
    }
}
