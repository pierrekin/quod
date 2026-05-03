"""quod.toml: required project-level config.

Every quod invocation needs a quod.toml. The CLI defaults `--config` to
`./quod.toml`; `quod init` creates one. There is no walk-up discovery and
no implicit defaults — config is explicit.

Paths inside quod.toml are resolved relative to the file's parent dir, so
`quod build -c /elsewhere/quod.toml` works regardless of CWD.

Schema:

    build_dir   = "build"
    proofs_dir  = "proofs"

    [build]
    profile = 2          # 0..3, LLVM -O level
    target  = ""         # triple; "" = host
    link    = true

    [enforce]
    axiom   = "trust"    # trust | verify
    witness = "trust"
    lattice = "trust"

    [[program]]
    name    = "hello"        # program identifier (used by --program / -p)
    version = "0.1.0"
    file    = "program.json" # path to the program JSON

      [[program.bin]]
      name  = "hello"        # output binary filename
      entry = "main"         # entry-point function in program.json

A workspace can list any number of `[[program]]` entries. Per-program
commands (`show`, `fn`, `claim`, ...) accept `--program / -p NAME`; if
exactly one program is configured the flag is optional.
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
class ProgramSpec:
    name: str
    version: str
    file: Path
    bins: tuple[Bin, ...] = ()


@dataclass(frozen=True)
class BuildConfig:
    profile: int = 2
    target: str = ""        # "" means host
    link: bool = True


@dataclass(frozen=True)
class LinkConfig:
    """Linker settings applied at the `clang object.o -o binary` step.

    `libraries` are bare names — e.g. ("m", "pthread") becomes `-lm -lpthread`.
    libc is always implicitly available (clang links it by default), so don't
    list "c". Project-wide; no per-program overrides.
    """
    libraries: tuple[str, ...] = ()


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
    programs: tuple[ProgramSpec, ...] = ()
    build_dir: Path = Path("build")
    proofs_dir: Path = Path("proofs")
    build: BuildConfig = field(default_factory=BuildConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    enforce: EnforceConfig = field(default_factory=EnforceConfig)
    # Directory the config was loaded from. Relative paths in the config
    # resolve against this — so build artifacts and program files are
    # anchored to quod.toml regardless of CWD.
    root: Path = field(default_factory=Path.cwd)

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    def select(self, name: str | None) -> ProgramSpec:
        """Pick a [[program]] by name. If name is None and exactly one program
        is configured, return it. Otherwise raise ValueError listing choices."""
        if name is None:
            if len(self.programs) == 1:
                return self.programs[0]
            if not self.programs:
                raise ValueError("no [[program]] entries declared")
            names = ", ".join(p.name for p in self.programs)
            raise ValueError(
                f"multiple programs ({names}); pass --program / -p NAME"
            )
        for p in self.programs:
            if p.name == name:
                return p
        names = ", ".join(p.name for p in self.programs) or "(none)"
        raise ValueError(f"no [[program]] named {name!r}; choices: {names}")


def load_config(path: Path) -> Config:
    """Load `path` as a quod.toml. Errors if the file is missing or invalid.

    At least one `[[program]]` is required for any build/inspection command,
    but `load_config` itself does not enforce that — `quod init` writes a
    quod.toml as part of project bootstrap.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no quod.toml here (run `quod init` to create one, "
            f"or pass --config PATH)"
        )

    raw = tomllib.loads(path.read_text())
    root = path.parent

    build_dir = Path(raw.get("build_dir", "build"))
    proofs_dir = Path(raw.get("proofs_dir", "proofs"))

    b = raw.get("build", {})
    build = BuildConfig(
        profile=int(b.get("profile", 2)),
        target=str(b.get("target", "")),
        link=bool(b.get("link", True)),
    )

    l = raw.get("link", {})
    libs_raw = l.get("libraries", [])
    if not isinstance(libs_raw, list):
        raise ValueError(f"{path}: [link] libraries must be a list of strings")
    link = LinkConfig(libraries=tuple(str(x) for x in libs_raw))

    e = raw.get("enforce", {})
    enforce = EnforceConfig(
        axiom=e.get("axiom"),
        witness=e.get("witness"),
        lattice=e.get("lattice"),
    )

    programs_raw = raw.get("program", [])
    if isinstance(programs_raw, dict):  # tolerate `[program]` (single)
        programs_raw = [programs_raw]
    programs: list[ProgramSpec] = []
    seen_names: set[str] = set()
    for entry in programs_raw:
        if "name" not in entry:
            raise ValueError(f"{path}: [[program]] entry missing required key `name`")
        if "file" not in entry:
            raise ValueError(
                f"{path}: [[program]] {entry['name']!r} missing required key `file`"
            )
        name = str(entry["name"])
        if name in seen_names:
            raise ValueError(f"{path}: duplicate [[program]] name {name!r}")
        seen_names.add(name)
        version = str(entry.get("version", "0.0.0"))
        file = Path(entry["file"])

        bins_raw = entry.get("bin", [])
        if isinstance(bins_raw, dict):
            bins_raw = [bins_raw]
        bins = tuple(
            Bin(name=str(bb["name"]), entry=str(bb.get("entry", bb["name"])))
            for bb in bins_raw
        )
        programs.append(ProgramSpec(name=name, version=version, file=file, bins=bins))

    return Config(
        programs=tuple(programs),
        build_dir=build_dir,
        proofs_dir=proofs_dir,
        build=build,
        link=link,
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
[build]
profile = 2

[[program]]
name    = "hello"
version = "0.1.0"
file    = "program.json"

  [[program.bin]]
  name  = "hello"
  entry = "main"
""",
    "guarded": """\
# `guarded` is a claim/proof playground — function `f` takes a parameter,
# so it can't be an entry point. Add a [[program.bin]] once you've written one.
[build]
profile = 2

[[program]]
name    = "guarded"
version = "0.1.0"
file    = "program.json"
""",
    "empty": """\
[build]
profile = 2

[[program]]
name    = "empty"
version = "0.1.0"
file    = "program.json"
""",
}


def starter_toml(template: str) -> str:
    """Return the quod.toml content `quod init -t TEMPLATE` should write."""
    return _STARTER_TOMLS[template]
