//! JSON-RPC client over a child quod-lsp process.
//!
//! Framing is handled by `lsp_server::Message::{read, write}` — the same
//! transport rust-analyzer uses. The client itself is synchronous: each
//! `request` writes a message and drains incoming traffic until the
//! matching response id arrives. Notifications/server-initiated requests
//! arriving in the meantime are dropped on the floor. That's fine for the
//! current hello-world flow; adding a worker thread + channel for
//! server-pushed traffic is the natural next step.
use std::io::BufReader;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use lsp_server::{Message, Notification, Request, RequestId, Response};
use serde_json::Value;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Protocol(String),
    Server { code: i32, message: String },
    Spawn(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Protocol(m) => write!(f, "protocol: {m}"),
            Error::Server { code, message } => write!(f, "server error {code}: {message}"),
            Error::Spawn(m) => write!(f, "spawn: {m}"),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i32,
}

impl Client {
    pub fn spawn(program: &str, args: &[&str]) -> Result<Self, Error> {
        // Server stderr to a log file so tracebacks survive the alt-screen
        // takeover. /tmp keeps it out of the working tree.
        let log = std::fs::File::create("/tmp/qui-server.log")?;
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log))
            .spawn()
            .map_err(|e| Error::Spawn(format!("could not start `{program}`: {e}")))?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout was piped"));
        Ok(Self { child, stdin, stdout, next_id: 0 })
    }

    pub fn request(&mut self, method: &str, params: Value) -> Result<Value, Error> {
        self.next_id += 1;
        let id: RequestId = self.next_id.into();
        let req = Request { id: id.clone(), method: method.to_string(), params };
        Message::Request(req).write(&mut self.stdin)?;
        loop {
            let msg = Message::read(&mut self.stdout)?
                .ok_or_else(|| Error::Protocol("server closed stdout".into()))?;
            match msg {
                Message::Response(Response { id: resp_id, result, error })
                    if resp_id == id =>
                {
                    if let Some(err) = error {
                        return Err(Error::Server { code: err.code, message: err.message });
                    }
                    return Ok(result.unwrap_or(Value::Null));
                }
                // Server-initiated request, notification, or response to
                // a different id — drop while we wait for our response.
                _ => continue,
            }
        }
    }

    pub fn notify(&mut self, method: &str, params: Value) -> Result<(), Error> {
        let notif = Notification { method: method.to_string(), params };
        Message::Notification(notif).write(&mut self.stdin)?;
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<(), Error> {
        let _ = self.request("shutdown", Value::Null)?;
        let _ = self.notify("exit", Value::Null);
        let _ = self.child.wait();
        Ok(())
    }
}
