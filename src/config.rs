use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rand::RngCore;

pub const DEFAULT_PORT: u16 = 7788;

/// Where memoryd keeps its files. Nothing here lives in the git repo.
#[derive(Debug, Clone)]
pub struct Paths {
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
}

impl Paths {
    /// `MEMORYD_HOME` overrides both directories (handy for tests and benchmarks).
    pub fn from_env() -> Result<Self> {
        if let Ok(root) = std::env::var("MEMORYD_HOME")
            && !root.is_empty()
        {
            let root = PathBuf::from(root);
            return Ok(Self { data_dir: root.join("data"), config_dir: root.join("config") });
        }
        let home = PathBuf::from(std::env::var("HOME").context("HOME is not set")?);
        Ok(Self {
            data_dir: home.join(".local/share/memoryd"),
            config_dir: home.join(".config/memoryd"),
        })
    }

    pub fn db(&self) -> PathBuf {
        self.data_dir.join("memories.db")
    }

    pub fn model_dir(&self) -> PathBuf {
        self.data_dir.join("model")
    }

    pub fn token(&self) -> PathBuf {
        self.config_dir.join("token")
    }
}

/// Reads the bearer token, creating a random one (mode 0600) on first use.
pub fn load_or_create_token(path: &Path) -> Result<String> {
    if path.exists() {
        return read_token(path);
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let token: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let mut file = match OpenOptions::new().write(true).create_new(true).mode(0o600).open(path) {
        Ok(f) => f,
        // Another process (e.g. `memoryd serve` starting while `memoryd token` runs) won the
        // race and wrote the file between our exists() check and create_new(); use its token.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return read_token(path),
        Err(e) => return Err(e.into()),
    };
    writeln!(file, "{token}")?;
    Ok(token)
}

fn read_token(path: &Path) -> Result<String> {
    let mode = fs::metadata(path)?.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        bail!("{} is readable by other users (mode {mode:o}); run chmod 600", path.display());
    }
    let token = fs::read_to_string(path)?.trim().to_owned();
    if token.len() < 32 {
        bail!("{} looks truncated", path.display());
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_created_once_with_private_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg/token");
        let first = load_or_create_token(&path).unwrap();
        let second = load_or_create_token(&path).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn token_with_loose_permissions_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        fs::write(&path, "x".repeat(64)).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_or_create_token(&path).is_err());
    }
}
