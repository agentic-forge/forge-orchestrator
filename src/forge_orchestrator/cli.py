"""CLI for Forge Orchestrator - LLM agent loop."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="orchestrator",
    help="LLM agent loop for Agentic Forge - connects UI to language models and MCP tools.",
    no_args_is_help=True,
)

console = Console()
error_console = Console(stderr=True)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from forge_orchestrator import __version__

        typer.echo(f"forge-orchestrator v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """LLM agent loop for Agentic Forge."""


@app.command()
def info() -> None:
    """Show orchestrator configuration and status."""
    from forge_orchestrator import __version__
    from forge_orchestrator.settings import settings

    console.print(f"[bold]Forge Orchestrator[/bold] v{__version__}")
    console.print()
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Armory URL: {settings.armory_url}")
    console.print(f"  Default Model: {settings.default_model}")
    console.print(f"  Server: {settings.host}:{settings.port}")
    console.print(f"  Conversations Dir: {settings.conversations_dir}")
    console.print(f"  Mock LLM Mode: {settings.mock_llm}")
    console.print(f"  Show Thinking: {settings.show_thinking}")
    console.print(f"  Heartbeat Interval: {settings.heartbeat_interval}s")
    console.print(f"  Tool Timeout Warning: {settings.tool_timeout_warning}s")

    # Check API key
    if settings.openrouter_api_key:
        console.print("  OpenRouter API Key: [green]configured[/green]")
    else:
        console.print("  OpenRouter API Key: [yellow]not configured[/yellow]")


@app.command()
def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Port to bind to"),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload for development"),
    ] = False,
) -> None:
    """Start the orchestrator server."""
    import uvicorn

    from forge_orchestrator.settings import settings

    actual_host = host or settings.host
    actual_port = port or settings.port

    console.print("[green]Starting Forge Orchestrator...[/green]")
    console.print(f"  Host: {actual_host}")
    console.print(f"  Port: {actual_port}")
    console.print(f"  Armory: {settings.armory_url}")
    console.print(f"  Model: {settings.default_model}")
    console.print()
    console.print(f"  API: http://{actual_host}:{actual_port}/")
    console.print(f"  Health: http://{actual_host}:{actual_port}/health")
    console.print()

    if settings.mock_llm:
        console.print("[yellow]Running in MOCK LLM mode - responses are simulated[/yellow]")
        console.print()

    uvicorn.run(
        "forge_orchestrator.server:app",
        host=actual_host,
        port=actual_port,
        reload=reload,
    )
