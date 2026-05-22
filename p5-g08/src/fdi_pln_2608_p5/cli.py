"""Punto de entrada del CLI."""

from fdi_pln_2608_p5.cli_app.app import app


def main() -> None:
    app()


__all__ = ["app", "main"]
