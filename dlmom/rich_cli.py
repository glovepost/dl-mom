#!/usr/bin/env python3
"""
DL-MoM Modern CLI v2 - Enhanced with Rich Output
=================================================

Features:
- Live updating progress with per-sample metrics
- Automatic anomaly detection and warnings
- Drift visualization relative to reference
- JSON + CSV export
- Quiet mode for CI/CD pipelines

Usage:
    from dlmom.rich_cli import DLMoMCLI, ExperimentResult, SampleResult
    
    cli = DLMoMCLI()
    cli.header(suite="A1", ...)
    result = cli.run_experiment_with_live_display(...)
    cli.summary()
    cli.export(output_dir)
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich import box
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterator
from enum import Enum
from pathlib import Path
import json
import csv
import time


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class SampleResult:
    """Per-sample result matching the logging schema."""
    sample_id: str
    correct: bool
    pred_answer: Optional[str] = None
    gold_answer: Optional[str] = None
    explicit_tokens: int = 0
    latent_steps: int = 0
    total_steps: int = 0
    wall_time_s: float = 0.0
    switches: int = 0
    peak_mem_gb: float = 0.0
    top_k_mass: float = 0.0
    failure_mode: Optional[str] = None
    seed: int = 0


@dataclass
class ExperimentResult:
    """Aggregated experiment result."""
    exp_id: str
    seed: int
    samples: list = field(default_factory=list)
    status: Status = Status.PENDING
    
    # Computed on finalization
    accuracy: float = 0.0
    mean_latent: float = 0.0
    mean_switches: float = 0.0
    timeouts: int = 0
    failures: int = 0
    total_time: float = 0.0
    kl_drift: Optional[float] = None
    cosine_drift: Optional[float] = None
    
    def finalize(self):
        """Compute aggregate metrics."""
        if not self.samples:
            self.status = Status.FAILED
            return
            
        n = len(self.samples)
        self.accuracy = 100 * sum(s.correct for s in self.samples) / n
        self.mean_latent = sum(s.latent_steps for s in self.samples) / n
        self.mean_switches = sum(s.switches for s in self.samples) / n
        self.timeouts = sum(1 for s in self.samples if s.failure_mode == "timeout")
        self.failures = sum(1 for s in self.samples if s.failure_mode is not None)
        self.total_time = sum(s.wall_time_s for s in self.samples)
        
        # Auto-detect issues
        if self.accuracy == 0 and self.failures == 0:
            self.status = Status.WARNING
        elif self.failures > n * 0.2:
            self.status = Status.WARNING
        else:
            self.status = Status.SUCCESS


class LiveExperimentDisplay:
    """Live updating display during experiment execution."""
    
    def __init__(self, console: Console):
        self.console = console
        self.reset()
    
    def update(self, sample: SampleResult):
        """Update live metrics with new sample."""
        self.current_metrics["total"] += 1
        if sample.correct:
            self.current_metrics["correct"] += 1
        self.current_metrics["latent_sum"] += sample.latent_steps
        self.current_metrics["switch_sum"] += sample.switches
        self.current_metrics["last_sample"] = sample
    
    def reset(self):
        """Reset for new experiment."""
        self.current_metrics = {
            "correct": 0,
            "total": 0,
            "latent_sum": 0,
            "switch_sum": 0,
            "last_sample": None,
        }
    
    def get_live_accuracy(self) -> str:
        """Get current running accuracy."""
        m = self.current_metrics
        n = m["total"]
        if n == 0:
            return "—"
        acc = 100 * m["correct"] / n
        return f"{acc:.0f}%"


class DLMoMCLI:
    """Main CLI interface for DL-MoM experiments."""
    
    def __init__(self, quiet: bool = False, no_color: bool = False):
        self.console = Console(
            stderr=True,
            force_terminal=not no_color,
            no_color=no_color,
        )
        self.quiet = quiet
        self.results: list[ExperimentResult] = []
        self.reference_id: Optional[str] = None
    
    def header(
        self,
        suite: str,
        benchmark: str,
        model: str,
        samples: int,
        seeds: list[int],
        experiments: list[str],
        device: str,
        reference: str = None,
        configs: list[dict] = None,
    ):
        """Print experiment header with parameter diff table."""
        if self.quiet:
            self.console.print(f"Suite={suite} bench={benchmark} n={samples}×{len(seeds)} exp={len(experiments)}")
            return
        
        self.reference_id = reference or experiments[0]
        
        # 1. Main Config Panel
        config_grid = Table(show_header=False, box=None, padding=(0, 2))
        config_grid.add_column("Key", style="dim")
        config_grid.add_column("Value")
        
        config_grid.add_row("Suite", f"[cyan bold]{suite}[/]")
        config_grid.add_row("Benchmark", f"[cyan]{benchmark}[/]")
        config_grid.add_row("Model", f"[green]{model}[/]")
        config_grid.add_row("Device", f"[yellow]{device}[/]")
        
        total = samples * len(seeds) * len(experiments)
        eta_per_sample = 5.0
        eta_min = (total * eta_per_sample) / 60
        
        config_grid.add_row("Workload", f"{samples} samples × {len(seeds)} seeds = [bold]{total}[/] total")
        config_grid.add_row("Est. Time", f"~{eta_min:.0f} min")

        self.console.print(Panel(
            config_grid,
            title="[bold white]🧪 DL-MoM Ablation Runner[/]",
            subtitle=f"[dim]{time.strftime('%Y-%m-%d %H:%M')}[/]",
            border_style="blue",
            box=box.DOUBLE,
        ))
        
        # 2. Experiment Parameter Diff Table
        if configs:
            # Identify varying keys
            all_keys = set().union(*[c.keys() for c in configs]) - {'id'}
            varying_keys = []
            for k in sorted(all_keys):
                vals = [str(c.get(k)) for c in configs]
                if len(set(vals)) > 1:
                    varying_keys.append(k)
            
            # Use specific important keys if none vary significantly (e.g. for single exp)
            if not varying_keys:
                varying_keys = [k for k in ["soft_mode", "gate", "comm", "kv"] if k in all_keys]

            if varying_keys:
                diff_table = Table(
                    title=f"[bold]Experiment Variants ({suite})[/]",
                    box=box.HEAVY,
                    header_style="bold cyan",
                    expand=True,
                )
                diff_table.add_column("ID", style="bold white", width=8)
                for k in varying_keys:
                    diff_table.add_column(k, justify="center")
                
                for c in configs:
                    row = [c.get('id', '?')]
                    for k in varying_keys:
                        val = c.get(k)
                        val_str = str(val) if val is not None else "[dim]null[/]"
                        # Highlight non-null values
                        if val is not None:
                            val_str = f"[cyan]{val_str}[/]"
                        row.append(val_str)
                    diff_table.add_row(*row)
                
                self.console.print(diff_table)
                self.console.print()
    
    def run_experiment_with_live_display(
        self,
        exp_id: str,
        seed: int,
        description: str,
        sample_generator: Iterator[SampleResult],
        total_samples: int,
    ) -> ExperimentResult:
        """Run experiment with live progress display."""
        
        result = ExperimentResult(exp_id=exp_id, seed=seed)
        result.status = Status.RUNNING
        
        live_display = LiveExperimentDisplay(self.console)
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[exp_id]}[/]"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[green]{task.fields[acc]}[/]"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        
        with progress:
            task = progress.add_task(
                "run",
                total=total_samples,
                exp_id=f"{exp_id} (seed={seed})",
                acc="—",
            )
            
            for sample in sample_generator:
                result.samples.append(sample)
                live_display.update(sample)
                
                # Update progress bar with live accuracy
                progress.update(
                    task,
                    advance=1,
                    acc=live_display.get_live_accuracy(),
                )
        
        result.finalize()
        self.results.append(result)
        
        # Print inline result
        self._print_result_line(result)
        
        return result
    
    def add_result(self, result: ExperimentResult):
        """Add a pre-computed result (for compatibility with existing runner)."""
        result.finalize()
        self.results.append(result)
        self._print_result_line(result)
    
    def _print_result_line(self, r: ExperimentResult):
        """Print single-line result summary."""
        icon = {
            Status.SUCCESS: "[green]✓[/]",
            Status.WARNING: "[yellow]⚠[/]",
            Status.FAILED: "[red]✗[/]",
        }.get(r.status, "?")
        
        # Color-code accuracy
        acc_color = "green" if r.accuracy >= 50 else "yellow" if r.accuracy >= 25 else "red"
        
        # Delta from reference
        ref = next((x for x in self.results if x.exp_id == self.reference_id and x.seed == r.seed), None)
        if ref and r.exp_id != self.reference_id:
            delta = r.accuracy - ref.accuracy
            delta_str = f"[green]+{delta:.1f}[/]" if delta > 0 else f"[red]{delta:.1f}[/]"
        else:
            delta_str = "[dim]ref[/]" if r.exp_id == self.reference_id else ""
        
        # Drift
        if r.kl_drift is not None:
            kl_color = "green" if r.kl_drift < 0.05 else "yellow" if r.kl_drift < 0.1 else "red"
            drift_str = f"KL:[{kl_color}]{r.kl_drift:.3f}[/]"
        else:
            drift_str = "[dim]KL:—[/]"
        
        line = (
            f"  {icon} [{acc_color}]{r.accuracy:5.1f}%[/] {delta_str:>8}  "
            f"latent:[cyan]{r.mean_latent:4.1f}[/]  "
            f"sw:[cyan]{r.mean_switches:.2f}[/]  "
            f"{drift_str}  "
            f"[dim]{r.total_time:.1f}s[/]"
        )
        
        if r.status == Status.WARNING:
            if r.accuracy == 0:
                line += "  [yellow bold]← 0% acc![/]"
            elif r.timeouts > 0:
                line += f"  [yellow]← {r.timeouts} timeouts[/]"
        
        self.console.print(line)
    
    def summary(self):
        """Print final summary table."""
        if self.quiet:
            for r in self.results:
                print(f"{r.exp_id},{r.accuracy:.1f},{r.mean_latent:.1f},{r.status.value}")
            return
        
        self.console.print()
        
        table = Table(
            title="[bold]📊 Results Summary[/]",
            box=box.ROUNDED,
            header_style="bold white on dark_blue",
        )
        
        table.add_column("", width=2)
        table.add_column("ID", style="bold")
        table.add_column("Seed", justify="right")
        table.add_column("Acc%", justify="right")
        table.add_column("Δ Ref", justify="right") 
        table.add_column("Latent", justify="right")
        table.add_column("Switches", justify="right")
        table.add_column("KL↓", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Fail", justify="right")
        
        for r in self.results:
            icon = "✓" if r.status == Status.SUCCESS else "⚠" if r.status == Status.WARNING else "✗"
            icon_style = "green" if r.status == Status.SUCCESS else "yellow" if r.status == Status.WARNING else "red"
            
            # Find reference for this seed
            ref = next((x for x in self.results if x.exp_id == self.reference_id and x.seed == r.seed), None)
            ref_acc = ref.accuracy if ref else 0
            
            # Delta
            if r.exp_id == self.reference_id:
                delta = "[dim]ref[/]"
            else:
                d = r.accuracy - ref_acc
                delta = f"[green]+{d:.1f}[/]" if d > 0 else f"[red]{d:.1f}[/]" if d < 0 else "[dim]0[/]"
            
            # Drift with color thresholds
            def fmt_drift(v, thresh_good=0.05, thresh_warn=0.1):
                if v is None:
                    return "[dim]—[/]"
                c = "green" if v < thresh_good else "yellow" if v < thresh_warn else "red"
                return f"[{c}]{v:.4f}[/]"
            
            # Failure rate
            n_samples = len(r.samples) if r.samples else 1
            fail_rate = f"{r.failures}/{n_samples}" if r.failures > 0 else "[dim]0[/]"
            
            table.add_row(
                f"[{icon_style}]{icon}[/]",
                r.exp_id,
                str(r.seed),
                f"{r.accuracy:.1f}",
                delta,
                f"{r.mean_latent:.1f}",
                f"{r.mean_switches:.2f}",
                fmt_drift(r.kl_drift),
                f"{r.total_time:.1f}s",
                fail_rate,
            )
        
        self.console.print(table)
        
        # Warnings panel
        warnings = [r for r in self.results if r.status == Status.WARNING]
        if warnings:
            warn_lines = []
            for r in warnings:
                if r.accuracy == 0:
                    warn_lines.append(f"[yellow]⚠ {r.exp_id} (seed={r.seed}):[/] 0% accuracy with 0 failures — likely config/compression bug")
                    if r.kl_drift and r.kl_drift > 0.5:
                        warn_lines.append(f"  [dim]→ High KL drift ({r.kl_drift:.3f}) suggests distribution collapse[/]")
                elif r.timeouts > 0:
                    n_samples = len(r.samples) if r.samples else 1
                    warn_lines.append(f"[yellow]⚠ {r.exp_id} (seed={r.seed}):[/] {r.timeouts} timeouts ({100*r.timeouts/n_samples:.0f}%)")
            
            self.console.print()
            self.console.print(Panel(
                "\n".join(warn_lines),
                title="[yellow bold]⚠ Warnings[/]",
                border_style="yellow",
            ))
    
    def export(self, output_dir: Path):
        """Export results to JSON and CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON with full detail
        json_data = []
        for r in self.results:
            json_data.append({
                "exp_id": r.exp_id,
                "seed": r.seed,
                "accuracy": r.accuracy,
                "mean_latent": r.mean_latent,
                "mean_switches": r.mean_switches,
                "timeouts": r.timeouts,
                "failures": r.failures,
                "total_time": r.total_time,
                "kl_drift": r.kl_drift,
                "cosine_drift": r.cosine_drift,
                "status": r.status.value,
                "samples": [asdict(s) for s in r.samples] if r.samples else [],
            })
        
        json_path = output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        
        # CSV summary
        csv_path = output_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "exp_id", "seed", "accuracy", "mean_latent", "mean_switches",
                "timeouts", "failures", "total_time", "kl_drift", "cosine_drift", "status"
            ])
            for r in self.results:
                writer.writerow([
                    r.exp_id, r.seed, r.accuracy, r.mean_latent, r.mean_switches,
                    r.timeouts, r.failures, r.total_time, r.kl_drift, r.cosine_drift, r.status.value
                ])
        
        self.console.print()
        self.console.print(f"[dim]Exported:[/] {json_path}")
        self.console.print(f"[dim]Exported:[/] {csv_path}")
    
    def footer(self, output_dir: str, total_time: float):
        """Print completion footer."""
        if self.quiet:
            print(f"Done in {total_time:.1f}s → {output_dir}")
            return
        
        success = sum(1 for r in self.results if r.status == Status.SUCCESS)
        warnings = sum(1 for r in self.results if r.status == Status.WARNING)
        failed = sum(1 for r in self.results if r.status == Status.FAILED)
        
        parts = [f"[green]{success} passed[/]"]
        if warnings:
            parts.append(f"[yellow]{warnings} warnings[/]")
        if failed:
            parts.append(f"[red]{failed} failed[/]")
        
        self.console.print()
        self.console.print(Panel(
            f"  Status: {', '.join(parts)}\n"
            f"  Time:   {total_time:.1f}s\n"
            f"  Output: [link=file://{output_dir}]{output_dir}[/]",
            title="[bold]✅ Complete[/]" if failed == 0 else "[bold red]❌ Complete with failures[/]",
            border_style="green" if failed == 0 else "red",
        ))


# Convenience factory
def create_cli(quiet: bool = False) -> DLMoMCLI:
    """Create a CLI instance."""
    return DLMoMCLI(quiet=quiet)
