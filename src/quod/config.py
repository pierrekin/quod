"""quod.toml: required project-level config.

Every quod invocation needs a quod.toml. The CLI defaults `--config` to
`./quod.toml`; `quod init` creates one. There is no walk-up discovery and
no implicit defaults — config is explicit.

Paths inside quod.toml are resolved relative to the file's parent dir, so
`quod build -c /elsewhere/quod.toml` works regardless of CWD.

Schema:

    program     = "program.json"   # required
    build_dir   = "build"
    proofs_dir  = "proofs"

    [build]
    profile = 2          # 0..3, LLVM -O level
    target  = ""         # triple; "" = host
    link    = true

    [[bin]]
    name  = "main"       # output binary filename
    entry = "main"       # entry-point function name in program.json

    [enforce]
    axiom   = "trust"    # trust | verify
    witness = "trust"
    lattice = "trust"
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


CONFIG_FILENAME = "quod.toml"


@dataclass(frozen=True)
class Bin:
    name: str
    entry: str


@dataclass(frozen=True)
class BuildConfig:
    profile: int = 2
    target: str = ""        # "" means host
    link: bool = True


@dataclass(frozen=True)
class EnforceConfig:
    axiom: str | None = None
    witness: str | None = None
    lattice: str | None = None

    def overrides(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.axiom is not None:
            out["axiom"] = self.axiom
        if self.witness is not None:
            out["witness"] = self.witness
        if self.lattice is not None:
            out["lattice"] = self.lattice
        return out


@dataclass(frozen=True)
class Config:
    program: Path
    build_dir: Path = Path("build")
    proofs_dir: Path = Path("proofs")
    build: BuildConfig = field(default_factory=BuildConfig)
    bins: tuple[Bin, ...] = ()
    enforce: EnforceConfig = field(default_factory=EnforceConfig)
    # Directory the config was loaded from. Relative paths in the config
    # resolve against this — so build artifacts and program files are
    # anchored to quod.toml regardless of CWD.
    root: Path = field(default_factory=Path.cwd)

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p


def load_config(path: Path) -> Config:
    """Load `path` as a quod.toml. Errors if the file is missing or invalid.

    `program` is required. `[[bin]]` entries are optional but `quod build`
    will error if there are none.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no quod.toml here (run `quod init` to create one, "
            f"or pass --config PATH)"
        )

    raw = tomllib.loads(path.read_text())
    root = path.parent

    if "program" not in raw:
        raise ValueError(f"{path}: missing required key `program`")
    program = Path(raw["program"])

    build_dir = Path(raw.get("build_dir", "build"))
    proofs_dir = Path(raw.get("proofs_dir", "proofs"))

    b = raw.get("build", {})
    build = BuildConfig(
        profile=int(b.get("profile", 2)),
        target=str(b.get("target", "")),
        link=bool(b.get("link", True)),
    )

    bins_raw = raw.get("bin", [])
    if isinstance(bins_raw, dict):  # tolerate `[bin]` (single) instead of `[[bin]]`
        bins_raw = [bins_raw]
    bins = tuple(
        Bin(name=str(b["name"]), entry=str(b.get("entry", b["name"])))
        for b in bins_raw
    )

    e = raw.get("enforce", {})
    enforce = EnforceConfig(
        axiom=e.get("axiom"),
        witness=e.get("witness"),
        lattice=e.get("lattice"),
    )

    return Config(
        program=program,
        build_dir=build_dir,
        proofs_dir=proofs_dir,
        build=build,
        bins=bins,
        enforce=enforce,
        root=root,
    )


def with_overrides(
    cfg: Config, *,
    profile: int | None = None,
    target: str | None = None,
    link: bool | None = None,
    enforce_axiom: str | None = None,
    enforce_witness: str | None = None,
    enforce_lattice: str | None = None,
) -> Config:
    """Apply CLI-flag overrides to a loaded Config."""
    new_build = replace(
        cfg.build,
        profile=profile if profile is not None else cfg.build.profile,
        target=target if target is not None else cfg.build.target,
        link=link if link is not None else cfg.build.link,
    )
    new_enforce = replace(
        cfg.enforce,
        axiom=enforce_axiom if enforce_axiom is not None else cfg.enforce.axiom,
        witness=enforce_witness if enforce_witness is not None else cfg.enforce.witness,
        lattice=enforce_lattice if enforce_lattice is not None else cfg.enforce.lattice,
    )
    return replace(cfg, build=new_build, enforce=new_enforce)


# ---------- Starter generation (used by `quod init`) ----------

_STARTER_TOMLS: dict[str, str] = {
    "hello": """\
program = "program.json"

[build]
profile = 2

[[bin]]
name = "hello"
entry = "main"
""",
    "guarded": """\
# `guarded` is a claim/proof playground — function `f` takes a parameter,
# so it can't be an entry point. Add a [[bin]] once you've written one.
program = "program.json"

[build]
profile = 2
""",
    "empty": """\
program = "program.json"

[build]
profile = 2
""",
}


def starter_toml(template: str) -> str:
    """Return the quod.toml content `quod init -t TEMPLATE` should write."""
    return _STARTER_TOMLS[template]
