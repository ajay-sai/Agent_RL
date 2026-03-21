#!/usr/bin/env python
"""Interactive demo for the Semantic Orchestrator system.

This demo allows you to:
- View system status and available backends
- Try sample queries from the test dataset
- Enter custom queries and see routing decisions, retrieval results, and synthesized answers
"""

import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from rich.text import Text

from semantic_orchestrator.config import load_config
from semantic_orchestrator.registry import SchemaRegistry
from semantic_orchestrator.router import RouterAgent, ACTIONS
from semantic_orchestrator.retrieval import create_retriever_for_backend
from semantic_orchestrator.synthesis import SynthesisAgent
from semantic_orchestrator.types import StorageBackend, BackendConfig

console = Console()


class DemoOrchestrator:
    """Main demo orchestrator that ties all components together."""

    def __init__(self):
        self.config = load_config()
        self.registry: SchemaRegistry | None = None
        self.router: RouterAgent | None = None
        self.retrievers: Dict[StorageBackend, Any] = {}
        self.synthesis: SynthesisAgent | None = None
        self.initialized = False
        self.sample_queries: List[str] = []

    def check_data_loaded(self) -> bool:
        """Check if registry.pkl exists and is valid."""
        registry_path = Path(self.config.data.processed_dir) / "registry.pkl"
        if not registry_path.exists():
            return False
        try:
            with open(registry_path, "rb") as f:
                pickle.load(f)
            return True
        except Exception:
            return False

    def load_data_if_needed(self) -> bool:
        """Load or prompt to load data."""
        if self.check_data_loaded():
            console.print("[green]✓ Data already loaded![/green]")
            return True

        console.print("[yellow]⚠ Data not found![/yellow]")
        console.print(f"Expected registry at: {self.config.data.processed_dir}/registry.pkl")

        if Confirm.ask("Would you like to load sample data now?"):
            console.print("\n[bold]Running data ingestion...[/bold]")
            import subprocess
            result = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "load_data.py")],
                capture_output=False,
            )
            if result.returncode == 0:
                console.print("[green]✓ Data loaded successfully![/green]")
                return True
            else:
                console.print("[red]✗ Data loading failed![/red]")
                return False
        else:
            console.print("[red]Cannot proceed without data. Exiting.[/red]")
            return False

    def load_sample_queries(self):
        """Load sample queries from test.jsonl."""
        test_path = Path(self.config.data.queries_test)
        if test_path.exists():
            try:
                import json
                with open(test_path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.sample_queries.append(data["query"])
                console.print(f"[green]✓ Loaded {len(self.sample_queries)} sample queries[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not load sample queries: {e}[/yellow]")
        else:
            console.print(f"[yellow]⚠ Sample queries not found at {test_path}[/yellow]")

    def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return

        console.print("\n[bold]Initializing system components...[/bold]")

        # Load registry
        registry_path = Path(self.config.data.processed_dir) / "registry.pkl"
        with open(registry_path, "rb") as f:
            self.registry = pickle.load(f)
        datasets = self.registry.list_datasets()
        console.print(f"[green]✓ Loaded registry with {len(datasets)} datasets: {', '.join(datasets)}[/green]")

        # Initialize router
        console.print("[bold]Initializing Router Agent...[/bold]")
        self.router = RouterAgent(registry=self.registry, device="cpu")
        console.print("[green]✓ Router initialized[/green]")

        # Initialize retrievers for each backend
        console.print("[bold]Initializing Retrievers...[/bold]")
        backends = [StorageBackend.VECTOR, StorageBackend.GRAPH, StorageBackend.SQL]
        for backend in backends:
            try:
                # Build BackendConfig from system config
                if backend == StorageBackend.VECTOR:
                    cfg = self.config.storage.vector
                    backend_config = BackendConfig(
                        backend_type=backend,
                        connection_string=cfg.persist_directory,
                        collection_name=cfg.collection_name,
                        extra_params={"embedding_model": cfg.embedding_model}
                    )
                elif backend == StorageBackend.GRAPH:
                    cfg = self.config.storage.graph
                    backend_config = BackendConfig(
                        backend_type=backend,
                        connection_string=cfg.uri,
                        collection_name=None,
                        username=cfg.username,
                        password=cfg.password
                    )
                elif backend == StorageBackend.SQL:
                    cfg = self.config.storage.sql
                    backend_config = BackendConfig(
                        backend_type=backend,
                        connection_string=cfg.database,
                        collection_name=None
                    )

                retriever = create_retriever_for_backend(backend, backend_config)
                # Test if backend has data
                count = retriever._store.count() if hasattr(retriever._store, 'count') else 0
                if count > 0:
                    self.retrievers[backend] = retriever
                    console.print(f"[green]  ✓ {backend.value}: {count} documents[/green]")
                else:
                    console.print(f"[yellow]  ⚠ {backend.value}: no data, skipping[/yellow]")
                    retriever.close()
            except Exception as e:
                console.print(f"[red]  ✗ {backend.value}: failed to initialize - {e}[/red]")

        if not self.retrievers:
            console.print("[red]✗ No backends available! Please check your data and configuration.[/red]")
            return False

        # Initialize synthesis agent
        console.print("[bold]Initializing Synthesis Agent...[/bold]")
        try:
            self.synthesis = SynthesisAgent()
            console.print("[green]✓ Synthesis Agent initialized[/green]")
        except ValueError as e:
            console.print(f"[yellow]⚠ Synthesis Agent: {e}[/yellow]")
            console.print("[yellow]  (Set OPENROUTER_API_KEY to enable answer synthesis)[/yellow]")

        self.initialized = True
        console.print("\n[bold green]System ready! You can now query.[/bold green]\n")
        return True

    def show_routing_decision(self, query: str, plan) -> Panel:
        """Display the router's decision in a panel."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Backend", style="cyan")
        table.add_column("Status", justify="center")

        for backend in plan.backends:
            status = "[green]Selected[/green]" if backend in self.retrievers else "[red]Unavailable[/red]"
            table.add_row(backend.value, status)

        content = f"[bold]Query:[/bold] {query}\n\n[bold]Selected Backends:[/bold] {', '.join(b.value for b in plan.backends)}\n\n"
        if len(plan.backends) > 1:
            content += f"[italic]Will query {len(plan.backends)} backends and synthesize results.[/italic]"
        else:
            content += f"[italic]Using specialized {plan.backends[0].value} backend.[/italic]"

        return Panel(content, title="[bold blue]Routing Decision[/bold blue]", border_style="blue")

    def show_retrieval_results(self, results_by_backend: Dict[StorageBackend, List]) -> Panel:
        """Display retrieval results from each backend in a table."""
        content = ""

        for backend, results in results_by_backend.items():
            if not results:
                content += f"\n[bold]{backend.value}:[/bold] [yellow]No results found[/yellow]\n"
                continue

            table = Table(title=f"{backend.value.capitalize()} Results ({len(results)} found)", show_header=True)
            table.add_column("Score", justify="right", style="cyan", width=6)
            table.add_column("Content Snippet", style="white")

            for r in results[:5]:  # Show top 5
                snippet = r.content[:100] + "..." if len(r.content) > 100 else r.content
                snippet = snippet.replace("\n", " ")
                table.add_row(f"{r.score:.3f}", snippet)

            if len(results) > 5:
                table.add_row("...", f"... and {len(results) - 5} more results")

            content += f"\n{table}\n"

        return Panel(content, title="[bold green]Retrieval Results[/bold green]", border_style="green")

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the full pipeline."""
        results_by_backend = {}
        selected_backends = []

        # Step 1: Routing decision
        plan = self.router.decide(query, eval_mode=True)
        selected_backends = plan.backends

        # Step 2: Retrieve from each available selected backend
        all_results = []
        for backend in selected_backends:
            if backend in self.retrievers:
                try:
                    results = self.retrievers[backend].search(query, k=10)
                    results_by_backend[backend] = results
                    all_results.extend(results)
                except Exception as e:
                    console.print(f"[red]Error retrieving from {backend.value}: {e}[/red]")
                    results_by_backend[backend] = []
            else:
                console.print(f"[yellow]Backend {backend.value} not available[/yellow]")
                results_by_backend[backend] = []

        # Step 3: Synthesize answer
        answer = ""
        if self.synthesis and all_results:
            try:
                answer = self.synthesis.process(query, all_results, top_k=10)
            except Exception as e:
                answer = f"[red]Error during synthesis: {e}[/red]"
        elif not all_results:
            answer = "[yellow]No results retrieved to synthesize an answer.[/yellow]"
        else:
            answer = "[yellow]Synthesis agent not available (missing OPENROUTER_API_KEY).[/yellow]"

        return {
            "query": query,
            "plan": plan,
            "results_by_backend": results_by_backend,
            "answer": answer,
            "total_results": len(all_results),
        }

    def run_interactive(self):
        """Run the interactive query loop."""
        console.print("[bold]Type your query and press Enter to process.[/bold]")
        console.print("[bold]Type 'quit', 'exit', or 'q' to exit.[/bold]")
        console.print("[bold]Type 'menu' to see sample queries.[/bold]\n")

        while True:
            query = Prompt.ask("\n[bold cyan]Enter your query[/bold cyan] (or 'quit' to exit)").strip()

            if query.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if query.lower() == "menu":
                self.show_sample_menu()
                continue

            if not query:
                continue

            console.print(f"\n[bold]Processing:[/bold] {query}\n")

            try:
                result = self.process_query(query)

                # Show routing decision
                console.print(self.show_routing_decision(query, result["plan"]))

                # Show retrieval results
                console.print()
                console.print(self.show_retrieval_results(result["results_by_backend"]))

                # Show synthesized answer
                console.print()
                answer_panel = Panel(
                    Markdown(result["answer"]),
                    title="[bold white]Synthesized Answer[/bold white]",
                    border_style="white",
                )
                console.print(answer_panel)

                # Show stats
                console.print(f"\n[dim]Total documents retrieved: {result['total_results']}[/dim]\n")
                console.print("─" * console.width)

            except Exception as e:
                console.print(f"[red]Error processing query: {e}[/red]")
                import traceback
                traceback.print_exc()

    def show_sample_menu(self):
        """Display menu of sample queries."""
        console.clear()
        console.print("[bold]Sample Queries Menu[/bold]\n")
        console.print("Select a query to try:\n")

        for i, query in enumerate(self.sample_queries, 1):
            console.print(f"  [cyan]{i}.[/cyan] {query}")

        console.print("\n  [cyan]0.[/cyan] Return to query input")
        console.print()

        choice = Prompt.ask("Select option", choices=[str(i) for i in range(len(self.sample_queries) + 1)], default="0")

        if choice == "0":
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.sample_queries):
                query = self.sample_queries[idx]
                console.print(f"\n[bold]Selected:[/bold] {query}\n")
                console.print(f"[dim]Processing...[/dim]\n")

                result = self.process_query(query)

                console.print(self.show_routing_decision(query, result["plan"]))
                console.print()
                console.print(self.show_retrieval_results(result["results_by_backend"]))
                console.print()
                answer_panel = Panel(
                    Markdown(result["answer"]),
                    title="[bold white]Synthesized Answer[/bold white]",
                    border_style="white",
                )
                console.print(answer_panel)

                console.print(f"\n[dim]Total documents retrieved: {result['total_results']}[/dim]\n")
                input("[bold]Press Enter to continue...[/bold]")
        except ValueError:
            console.print("[red]Invalid selection[/red]")


def main():
    """Main entry point."""
    # Welcome message
    welcome_text = """
[bold white]Semantic Orchestrator Demo[/bold white]

This interactive demo showcases the semantic data orchestration system that:
• Uses RL to route queries to optimal storage backends
• Retrieves relevant data from Vector (Chroma), Graph (Neo4j), and SQL (SQLite) stores
• Synthesizes answers from multiple sources using AI

[bold cyan]Features:[/bold cyan]
• View routing decisions and reasoning
• See retrieval results from each backend with relevance scores
• Get synthesized answers with citations
• Try sample queries or enter your own
"""
    console.print(Panel(welcome_text, border_style="cyan", expand=False))

    # Create orchestrator
    demo = DemoOrchestrator()

    # Check data
    if not demo.load_data_if_needed():
        sys.exit(1)

    # Load sample queries
    demo.load_sample_queries()

    # Initialize components
    if not demo.initialize():
        console.print("[red]Failed to initialize. Exiting.[/red]")
        sys.exit(1)

    # Run interactive loop
    demo.run_interactive()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
