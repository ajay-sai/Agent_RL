"""Command-line interface for semantic orchestrator."""
import argparse
import sys
import subprocess
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Semantic Orchestrator - RL Router Training & Inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Load data command
    load_parser = subparsers.add_parser("load-data", help="Load CSV data into storage backends")
    load_parser.add_argument(
        "--dataset",
        type=str,
        default="sample_sales",
        help="Dataset name (looks in data/raw/{dataset}.csv)",
    )
    load_parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file (overrides --dataset)",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a single query")
    query_parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Query string (if not provided, reads from stdin)",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of documents to retrieve per backend (default: 10)",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the router policy")
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config)",
    )
    train_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    train_parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Reward scale factor (overrides config)",
    )
    train_parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name (overrides config)",
    )
    train_parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (overrides config)",
    )
    train_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of documents to retrieve per backend (default: 10)",
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Launch interactive demo")
    demo_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Load config
    config = load_config(args.config)

    if args.command == "load-data":
        # Run the load_data script
        script_path = Path(__file__).parent.parent.parent / "scripts" / "load_data.py"
        if not script_path.exists():
            print(f"Error: load_data.py not found at {script_path}", file=sys.stderr)
            return 1

        cmd = [sys.executable, str(script_path)]
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        if args.csv:
            cmd.extend(["--csv", args.csv])

        result = subprocess.run(cmd)
        return result.returncode

    elif args.command == "query":
        from .orchestrator import QueryOrchestrator

        query = args.query
        if not query:
            print("Enter query: ", end="")
            query = sys.stdin.readline().strip()

        if not query:
            print("Error: empty query", file=sys.stderr)
            return 1

        try:
            with QueryOrchestrator(config_path=args.config) as orch:
                answer = orch.query(query, top_k=args.k)
                print(answer)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Hint: Run 'semantic-orchestrator load-data' first.", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.command == "train":
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Initialize registry
        from .registry import SchemaRegistry
        registry = SchemaRegistry()
        registry_path = Path(config.data.processed_dir) / "registry.pkl"
        if registry_path.exists():
            import pickle
            with open(registry_path, "rb") as f:
                registry = pickle.load(f)
            print(f"Loaded registry with datasets: {registry.list_datasets()}")
        else:
            print("Warning: Registry not found. Some actions may be masked.", file=sys.stderr)

        # Initialize router
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        from .router import RouterAgent
        router = RouterAgent(config_path=args.config, registry=registry, device=device)

        # Prepare WandB arguments if enabled
        wandb_project = None
        wandb_entity = None
        if args.wandb:
            wandb_project = args.wandb_project or config.logging.wandb_project or "semantic-orchestrator"
            wandb_entity = args.wandb_entity or config.logging.wandb_entity

        # Train
        router.train(
            epochs=args.epochs,
            learning_rate=args.lr,
            reward_scale=args.reward_scale,
            seed=args.seed,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            k=args.k,
        )

    elif args.command == "demo":
        # Run the demo script
        demo_script = Path(__file__).parent.parent.parent / "scripts" / "demo.py"
        if not demo_script.exists():
            print(f"Error: demo.py not found at {demo_script}", file=sys.stderr)
            return 1

        result = subprocess.run([sys.executable, str(demo_script)])
        return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
