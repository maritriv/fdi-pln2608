"""Definición de la aplicación Typer principal."""

import typer

from fdi_pln_2608_p5.cli_app.commands import register_commands
from fdi_pln_2608_p5.cli_app.interactive import run_interactive_menu


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="P5 PLN 2608: mini LLM Transformer causal, tokenización BPE y NER BIO.",
    invoke_without_command=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Abre el menú interactivo si no se indica un subcomando."""

    if ctx.invoked_subcommand is None:
        run_interactive_menu()
        raise typer.Exit()


register_commands(app)
