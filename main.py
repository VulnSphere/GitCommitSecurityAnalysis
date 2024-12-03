import argparse
import logging
import json
from rich.console import Console
from rich.table import Table
from git_security_analyzer import GitSecurityAnalyzer

logging.basicConfig(level=logging.INFO)
console = Console()

def display_analysis(analysis_results):
    """Display analysis results in a formatted table."""
    table = Table(title="Git Security Analysis Results")
    
    table.add_column("Commit Hash", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Security Issues", style="red")
    table.add_column("Summary", style="green")
    
    for result in analysis_results:
        commit_type = []
        if result.is_documentation:
            commit_type.append("Documentation")
        if result.is_dependency_update:
            commit_type.append("Dependency")
        if not commit_type:
            commit_type.append("Code")
            
        security_issues = len(result.security_issues)
        
        table.add_row(
            result.commit_hash[:8],
            ", ".join(commit_type),
            str(security_issues),
            result.analysis_summary
        )
    
    console.print(table)
    
    # Print detailed security issues
    for result in analysis_results:
        if result.security_issues:
            console.print(f"\n[bold red]Security Issues in commit {result.commit_hash[:8]}:[/bold red]")
            for issue in result.security_issues:
                console.print(f"[red]- Severity: {issue.severity}[/red]")
                console.print(f"  Description: {issue.description}")
                if issue.code_snippet:
                    console.print(f"  Code: {issue.code_snippet}")
                if issue.recommendation:
                    console.print(f"  Recommendation: {issue.recommendation}")
                console.print("")


def main():
    parser = argparse.ArgumentParser(description="Analyze Git repository commits for security issues")
    parser.add_argument("repo_path", help="Path to the Git repository")
    parser.add_argument("--commits", type=int, default=10, help="Number of commits to analyze (default: 10)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--llm", choices=["claude", "gpt", "ollama"], default="claude", 
                      help="LLM backend to use (default: claude)")
    args = parser.parse_args()

    try:
        # Environment variables for models and base URLs are handled in initialize_llm
        analyzer = GitSecurityAnalyzer(args.repo_path, llm_type=args.llm)
        results = analyzer.analyze_commits(args.commits)
        
        # Display results
        display_analysis(results)
        
        # Save to JSON if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump([r.model_dump() for r in results], f, indent=2)
                console.print(f"\nResults saved to {args.output}")
                
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
