"""provider sub-app — inspect registered claim providers."""

from __future__ import annotations

import typer

from quod.cli.app import provider_app
from quod.cli.output import _theme
from quod.predicate.providers import all_providers
from quod.render import Span, paint


@provider_app.command("ls")
def provider_ls() -> None:
    """List registered claim providers (regimes + supported modes)."""
    providers = all_providers()
    if not providers:
        typer.echo("(no providers registered)")
        return
    theme = _theme()
    name_w = max(len(p.name) for p in providers.values())
    regime_w = max(len(p.regime) for p in providers.values())
    for p in providers.values():
        modes = "+".join(p.modes) if p.modes else "(none)"
        name_pad = " " * (name_w - len(p.name))
        regime_pad = " " * (regime_w - len(p.regime))
        typer.echo(paint((
            Span(p.name, "fn_name"), Span(name_pad, "ws"), Span("  ", "ws"),
            Span("regime=", "meta_label"), Span(p.regime, "meta_value"),
            Span(regime_pad, "ws"), Span("  ", "ws"),
            Span("modes=", "meta_label"), Span(modes, "meta_value"),
        ), theme))
        typer.echo(paint((Span(f"  {p.description}", "comment"),), theme))
