"""CLI entry point for StudyRAG."""
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from src.rag.pipeline import RAGPipeline

app = typer.Typer(name="StudyRAG", help="RAG-based study assistant for course materials")
console = Console()


@app.command()
def ingest(
    docs_dir: Path = typer.Option(
        "./data/documents",
        "--dir", "-d",
        help="Directory containing PDF/PPTX files"
    )
):
    """Ingest course materials into vector store."""
    pipeline = RAGPipeline()
    pipeline.ingest(docs_dir)
    stats = pipeline.stats()
    console.print(f"[green]Ingested {stats['document_count']} chunks[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve")
):
    """Ask a question about your course materials."""
    pipeline = RAGPipeline()
    # Override top_k for this query
    pipeline.cfg["TOP_K"] = top_k

    with console.status("[bold green]Thinking..."):
        result = pipeline.query(question)

    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))

    if result["sources"]:
        console.print("\n[bold]Sources:[/bold]")
        for src in result["sources"]:
            console.print(f"  • {src['source']} (chunk {src['chunk_id']}, score: {src['score']})")


@app.command()
def chat(
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve")
):
    """Interactive chat mode."""
    pipeline = RAGPipeline()
    pipeline.cfg["TOP_K"] = top_k

    console.print(Panel("StudyRAG Chat Mode - Type 'exit' to quit", border_style="blue"))

    while True:
        question = console.input("\n[bold cyan]You:[/bold cyan] ")
        if question.lower() in {"exit", "quit", "q"}:
            break

        with console.status("[bold green]Thinking..."):
            result = pipeline.query(question)

        console.print(Panel(Markdown(result["answer"]), title="Assistant", border_style="green"))

        if result["sources"]:
            console.print("[dim]Sources:[/dim]")
            for src in result["sources"]:
                console.print(f"  [dim]• {src['source']} (chunk {src['chunk_id']}, score: {src['score']})[/dim]")


@app.command()
def stats():
    """Show vector store statistics."""
    pipeline = RAGPipeline()
    stats = pipeline.stats()
    console.print(f"[bold]Documents in store:[/bold] {stats['document_count']}")


@app.command()
def clear():
    """Clear all ingested documents."""
    pipeline = RAGPipeline()
    pipeline.clear()


if __name__ == "__main__":
    app()