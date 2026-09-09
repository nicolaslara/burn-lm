//! Fetching a model's files out of a Hugging Face repo.
//!
//! The existing downloader (`pretrained::downloader`) fetches one URL per role into
//! `~/.cache/llama/<name>/`, buffers the whole response in memory, and panics on a network error.
//! That is fine for the published `.mpk` checkpoints, which are a fixed pair of URLs. A canonical
//! repo is a different shape: several files named relative to a repo id and a revision, one of them
//! a couple of gigabytes, and the interesting ones (`tokenizer.model` in a repo that ships
//! `tokenizer.json` instead) legitimately absent. So this is its own fetcher — streaming to disk,
//! resuming a partial file with a `Range` request, retrying a dropped connection, and telling
//! "not there" apart from "could not get there".
//!
//! Files land in `~/.cache/llama/hf/<owner>__<name>[@<revision>]/`, beside the existing
//! per-model directories rather than inside one, because the cache key here is a repo id and not
//! one of our five names.

use std::fs::{create_dir_all, OpenOptions};
use std::path::PathBuf;
use std::time::Duration;

/// How many times a single file is attempted before giving up. Each attempt resumes where the last
/// one stopped, so a large file that keeps getting cut off still makes progress.
const ATTEMPTS: usize = 8;

/// How long to wait between attempts. Matches the delay the download instructions in the repo use.
const RETRY_DELAY: Duration = Duration::from_secs(5);

/// One Hugging Face model repo at one revision.
#[derive(Debug, Clone)]
pub struct HfRepo {
    repo_id: String,
    revision: String,
}

impl HfRepo {
    /// A repo at its default branch.
    ///
    /// `repo_id` is the `owner/name` form, the same string `vllm serve` takes.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: "main".to_string(),
        }
    }

    /// Pin the repo to a branch, tag, or commit sha.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// The repo id this was built from, for error messages.
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// Where this repo's files are cached.
    ///
    /// The repo id's `/` cannot go into a directory name, so it becomes `__`. A pinned revision is
    /// part of the key — two revisions of one repo are two different sets of weights and must not
    /// share a cache entry — while the default branch is left out so the common case reads as the
    /// repo it is.
    pub fn cache_dir(&self) -> PathBuf {
        let mut name = self.repo_id.replace('/', "__");
        if self.revision != "main" {
            name.push('@');
            name.push_str(&self.revision);
        }
        dirs::home_dir()
            .expect("should be able to get the home directory")
            .join(".cache")
            .join("llama")
            .join("hf")
            .join(name)
    }

    /// Fetch a file the model cannot be loaded without, returning its local path.
    pub fn file(&self, name: &str) -> Result<PathBuf, String> {
        match self.fetch(name)? {
            Some(path) => Ok(path),
            None => Err(format!(
                "{} has no {name} at revision {}",
                self.repo_id, self.revision
            )),
        }
    }

    /// Fetch a file the model may or may not ship, returning `None` when the repo does not have it.
    ///
    /// Only a 404 is `None`. A refused or unreachable request is still an error, so a network
    /// problem is never mistaken for a repo that simply lays its files out differently.
    pub fn optional_file(&self, name: &str) -> Result<Option<PathBuf>, String> {
        self.fetch(name)
    }

    fn url(&self, name: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{name}",
            self.repo_id, self.revision
        )
    }

    /// Download `name` into the cache directory unless it is already there.
    ///
    /// The bytes stream into `<name>.part` and are renamed into place only once the whole file has
    /// arrived and its length matches what the server said to expect, so a cut-off download can
    /// never be picked up as a complete file. `.part` is deliberately left behind on failure: the
    /// next attempt asks for `bytes=<what we have>-` and carries on from there.
    fn fetch(&self, name: &str) -> Result<Option<PathBuf>, String> {
        let dir = self.cache_dir();
        let target = dir.join(name);
        if target.exists() {
            return Ok(Some(target));
        }
        create_dir_all(&dir).map_err(|err| format!("could not create {}: {err}", dir.display()))?;

        let url = self.url(name);
        let partial = dir.join(format!("{name}.part"));
        let client = reqwest::blocking::Client::builder()
            // No overall timeout: this call is a multi-gigabyte download, and a deadline on the
            // whole request would fail the big file and nothing else. A host that cannot be
            // reached at all is still caught quickly, by the connect timeout.
            .timeout(None)
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|err| format!("could not build an HTTP client: {err}"))?;

        let mut last_error = String::new();
        for attempt in 1..=ATTEMPTS {
            let have = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

            let mut request = client.get(&url);
            if have > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
            }
            // A gated repo needs a token. We only read one if the environment already offers it —
            // this never goes looking for a credentials file.
            if let Ok(token) = std::env::var("HF_TOKEN") {
                if !token.is_empty() {
                    request = request.bearer_auth(token);
                }
            }

            match request.send() {
                Ok(mut response) => {
                    let status = response.status();
                    if status == reqwest::StatusCode::NOT_FOUND {
                        let _ = std::fs::remove_file(&partial);
                        return Ok(None);
                    }
                    if status == reqwest::StatusCode::UNAUTHORIZED
                        || status == reqwest::StatusCode::FORBIDDEN
                    {
                        return Err(format!(
                            "{url} is gated ({status}); set HF_TOKEN to an account that has \
                             accepted its licence, or use an open mirror of the same weights"
                        ));
                    }
                    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                        // The partial file is longer than the file on the server: it belongs to
                        // something else. Start over.
                        let _ = std::fs::remove_file(&partial);
                        last_error = format!("{url} rejected a resume from byte {have}");
                        continue;
                    }
                    if !status.is_success() {
                        last_error = format!("{url} returned {status}");
                        std::thread::sleep(RETRY_DELAY);
                        continue;
                    }

                    // A server that honored the range appends; one that ignored it (200 with the
                    // whole body) starts the file again.
                    let resuming = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
                    let expected =
                        response.content_length().map(
                            |len| {
                                if resuming {
                                    have + len
                                } else {
                                    len
                                }
                            },
                        );

                    let mut options = OpenOptions::new();
                    options.create(true).write(true);
                    if resuming {
                        options.append(true);
                    } else {
                        options.truncate(true);
                    }
                    let mut file = options
                        .open(&partial)
                        .map_err(|err| format!("could not open {}: {err}", partial.display()))?;

                    println!("downloading {url}");
                    if let Err(err) = std::io::copy(&mut response, &mut file) {
                        last_error = format!("{url} download interrupted: {err}");
                        eprintln!("{last_error} (attempt {attempt}/{ATTEMPTS}), resuming");
                        std::thread::sleep(RETRY_DELAY);
                        continue;
                    }
                    drop(file);

                    let written = std::fs::metadata(&partial)
                        .map(|m| m.len())
                        .map_err(|err| format!("could not stat {}: {err}", partial.display()))?;
                    if let Some(expected) = expected {
                        if written != expected {
                            last_error = format!("{url} gave {written} of {expected} bytes");
                            eprintln!("{last_error} (attempt {attempt}/{ATTEMPTS}), resuming");
                            std::thread::sleep(RETRY_DELAY);
                            continue;
                        }
                    }

                    std::fs::rename(&partial, &target).map_err(|err| {
                        format!("could not move {} into place: {err}", partial.display())
                    })?;
                    return Ok(Some(target));
                }
                Err(err) => {
                    last_error = format!("{url}: {err}");
                    eprintln!("{last_error} (attempt {attempt}/{ATTEMPTS}), retrying");
                    std::thread::sleep(RETRY_DELAY);
                }
            }
        }

        Err(format!(
            "could not download {name} from {} after {ATTEMPTS} attempts. Last error: {last_error}",
            self.repo_id
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_keys_on_repo_id_and_pinned_revision() {
        let repo = HfRepo::new("unsloth/Llama-3.2-1B-Instruct");
        assert!(repo
            .cache_dir()
            .ends_with("llama/hf/unsloth__Llama-3.2-1B-Instruct"));

        let pinned = repo.clone().with_revision("abc123");
        assert!(pinned
            .cache_dir()
            .ends_with("llama/hf/unsloth__Llama-3.2-1B-Instruct@abc123"));
    }

    #[test]
    fn urls_name_the_revision() {
        let repo = HfRepo::new("unsloth/Llama-3.2-1B-Instruct");
        assert_eq!(
            repo.url("config.json"),
            "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/config.json"
        );
    }
}
