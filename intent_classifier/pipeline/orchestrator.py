"""
Pipeline Orchestrator

This module provides a Pythonic way to run the training pipeline by calling
the main functions from each pipeline script directly, rather than using subprocess.

This makes the pipeline:
- Easier to test
- Easier to debug
- Easier to reuse in other contexts
- More Pythonic

The scripts can still be run directly for backward compatibility.
"""

import sys
from pathlib import Path
from typing import Any

from intent_classifier.utils.paths import get_repo_root
from intent_classifier.utils.seed import set_global_seed
from intent_classifier.utils.warnings_config import suppress_pydantic_warnings


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline steps as Python functions."""

    def __init__(self, repo_root: Path | None = None):
        """Initialize the orchestrator.

        Args:
            repo_root: Repository root directory. If None, will be discovered automatically.
        """
        if repo_root is None:
            repo_root = get_repo_root()
        self.repo_root = Path(repo_root).resolve()
        self.script_dir = self.repo_root / "scripts" / "pipeline"

        # Ensure repo_root is in sys.path for config imports
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # Suppress verbose Pydantic warnings
        suppress_pydantic_warnings()

    def _import_script_module(self, script_name: str) -> Any:
        """Import a pipeline script as a module.

        Args:
            script_name: Name of the script file (e.g., "00_data_loading.py")

        Returns:
            Imported module

        Raises:
            ImportError: If the script cannot be imported
            FileNotFoundError: If the script file does not exist
        """
        script_path = self.script_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Import the script as a module
        # We need to handle the import carefully since scripts are not packages
        import importlib.util

        # Create a unique module name to avoid conflicts
        module_name = f"pipeline_{script_name.replace('.py', '').replace('-', '_')}"

        # Check if module is already loaded (avoid re-importing)
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {script_name}")

        module = importlib.util.module_from_spec(spec)
        # Add repo_root to sys.path if not already there (scripts do this)
        # Note: Scripts also do this, so we ensure it's done before executing
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # Execute the module (this will run all top-level code including imports)
        spec.loader.exec_module(module)

        # Store in sys.modules to allow re-import if needed
        sys.modules[module_name] = module

        return module

    def run_step(self, script_name: str) -> bool:
        """Run a single pipeline step.

        Args:
            script_name: Name of the script file (e.g., "00_data_loading.py")

        Returns:
            True if successful, False otherwise
        """
        try:
            module = self._import_script_module(script_name)
            if hasattr(module, "main"):
                module.main()
                return True
            else:
                print(f"⚠️  Warning: {script_name} does not have a main() function")
                return False
        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_all(
        self,
        steps: list[str] | None = None,
        skip_tuning: bool = False,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """Run all pipeline steps in sequence.

        Args:
            steps: Optional list of script names to run. If None, runs all default steps.
            skip_tuning: If True, skip hyperparameter tuning step.
            config: Optional config dict. If None, will be loaded automatically.

        Returns:
            True if all steps completed successfully, False otherwise
        """
        if steps is None:
            steps = [
                "00_data_loading.py",
                "01_exploratory_analysis.py",
                "02_build_embeddings.py",
                "03_model_training.py",
                "04_model_prediction.py",
                "05_model_evaluation.py",
                "06_finalize_run.py",
            ]

        # Load config to get seed
        if config is None:
            try:
                from intent_classifier.utils.config_loader import load_config

                config = load_config()
            except Exception as e:
                print(f"⚠️  Warning: Could not load config: {e}")
                print("   Using default seed: 42")
                config = {}

        seed = config.get("general", {}).get("seed", 42)
        set_global_seed(seed)

        print("=" * 60)
        print("Running Complete Training Pipeline")
        print("=" * 60)
        print(f"Config: {config.get('_config_file', 'default')}")
        print(f"Seed: {seed}")
        print()

        # Run hyperparameter tuning if requested (before training)
        if not skip_tuning:
            print("\n" + "=" * 60)
            print("Step 0: Hyperparameter Tuning")
            print("=" * 60)
            try:
                from intent_classifier.hparam.tune import run_tuning

                best_params = run_tuning(config)
                if best_params:
                    print(f"\n✅ Tuning complete. Best parameters for {len(best_params)} model(s)")
                else:
                    print("\n⚠️  No hyperparameters returned from tuning")
            except ImportError:
                print("⚠️  Warning: Hyperparameter tuning module not available")
                print('   Install with: uv pip install "ray[tune]>=2.0,<3"')
            except Exception as e:
                print(f"\n❌ Hyperparameter tuning failed: {e}")
                print("   Continuing with default hyperparameters...")
                import traceback

                traceback.print_exc()

        # Run pipeline steps
        all_success = True
        for i, script_name in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] Running {script_name}...")
            print("-" * 60)

            success = self.run_step(script_name)
            if success:
                print(f"✅ {script_name} completed successfully")
            else:
                print(f"❌ {script_name} failed")
                all_success = False
                break  # Stop on first failure

        if all_success:
            print("\n" + "=" * 60)
            print("✅ All pipeline scripts completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ Pipeline failed")
            print("=" * 60)

        return all_success
