#!/usr/bin/env python3
"""
Security scanning script for refunc.

This script runs comprehensive security scanning using multiple tools:
- bandit: Python security linting
- safety: Dependency vulnerability scanning  
- semgrep: Static analysis security scanning
- pip-audit: Python package vulnerability scanning
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json


class SecurityScanner:
    """Comprehensive security scanner for Python projects."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        
    def run_bandit(self) -> Dict[str, Any]:
        """Run bandit security linting."""
        print("🔍 Running bandit security linting...")
        
        try:
            cmd = [
                "bandit", "-r", str(self.project_root / "refunc"),
                "-f", "json",
                "-c", str(self.project_root / ".bandit")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                bandit_results = json.loads(result.stdout) if result.stdout else {}
                print(f"✅ Bandit scan completed - {len(bandit_results.get('results', []))} issues found")
                return {"status": "success", "issues": bandit_results.get('results', [])}
            else:
                print(f"⚠️  Bandit scan completed with warnings - return code {result.returncode}")
                # Bandit returns non-zero when issues are found, but output is still valid
                bandit_results = json.loads(result.stdout) if result.stdout else {}
                return {"status": "issues_found", "issues": bandit_results.get('results', [])}
                
        except FileNotFoundError:
            print("❌ Bandit not installed. Install with: pip install bandit")
            return {"status": "error", "message": "bandit not found"}
        except Exception as e:
            print(f"❌ Bandit scan failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_safety(self) -> Dict[str, Any]:
        """Run safety dependency vulnerability scanning."""
        print("🔍 Running safety dependency scanning...")
        
        try:
            cmd = ["safety", "check", "--json", "--ignore", ""]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Safety scan completed - no vulnerabilities found")
                return {"status": "success", "vulnerabilities": []}
            else:
                # Parse JSON output if available
                try:
                    safety_results = json.loads(result.stdout) if result.stdout else []
                    print(f"⚠️  Safety scan found {len(safety_results)} vulnerabilities")
                    return {"status": "vulnerabilities_found", "vulnerabilities": safety_results}
                except json.JSONDecodeError:
                    print(f"⚠️  Safety scan completed with warnings")
                    return {"status": "warnings", "message": result.stdout or result.stderr}
                    
        except FileNotFoundError:
            print("❌ Safety not installed. Install with: pip install safety")
            return {"status": "error", "message": "safety not found"}
        except Exception as e:
            print(f"❌ Safety scan failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_semgrep(self) -> Dict[str, Any]:
        """Run semgrep static analysis security scanning."""
        print("🔍 Running semgrep security analysis...")
        
        try:
            cmd = [
                "semgrep", "--config", str(self.project_root / ".semgrep.yml"),
                "--json", str(self.project_root / "refunc")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                semgrep_results = json.loads(result.stdout) if result.stdout else {}
                issues = semgrep_results.get('results', [])
                print(f"✅ Semgrep scan completed - {len(issues)} issues found")
                return {"status": "success", "issues": issues}
            else:
                print(f"⚠️  Semgrep scan completed with issues")
                try:
                    semgrep_results = json.loads(result.stdout) if result.stdout else {}
                    issues = semgrep_results.get('results', [])
                    return {"status": "issues_found", "issues": issues}
                except json.JSONDecodeError:
                    return {"status": "error", "message": result.stderr}
                    
        except FileNotFoundError:
            print("❌ Semgrep not installed. Install with: pip install semgrep")
            return {"status": "error", "message": "semgrep not found"}
        except Exception as e:
            print(f"❌ Semgrep scan failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_pip_audit(self) -> Dict[str, Any]:
        """Run pip-audit package vulnerability scanning."""
        print("🔍 Running pip-audit package scanning...")
        
        try:
            cmd = ["pip-audit", "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Pip-audit scan completed - no vulnerabilities found")
                return {"status": "success", "vulnerabilities": []}
            else:
                # Parse JSON output if available
                try:
                    audit_results = json.loads(result.stdout) if result.stdout else []
                    print(f"⚠️  Pip-audit found {len(audit_results)} vulnerabilities")
                    return {"status": "vulnerabilities_found", "vulnerabilities": audit_results}
                except json.JSONDecodeError:
                    print(f"⚠️  Pip-audit completed with warnings")
                    return {"status": "warnings", "message": result.stdout or result.stderr}
                    
        except FileNotFoundError:
            print("❌ Pip-audit not installed. Install with: pip install pip-audit")
            return {"status": "error", "message": "pip-audit not found"}
        except Exception as e:
            print(f"❌ Pip-audit scan failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans."""
        print("🚀 Starting comprehensive security scan...\n")
        
        self.results = {
            "bandit": self.run_bandit(),
            "safety": self.run_safety(), 
            "semgrep": self.run_semgrep(),
            "pip_audit": self.run_pip_audit()
        }
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        if not self.results:
            self.run_all_scans()
        
        # Count issues
        total_issues = 0
        critical_issues = 0
        
        # Bandit issues
        bandit_issues = len(self.results.get("bandit", {}).get("issues", []))
        total_issues += bandit_issues
        
        # Safety vulnerabilities
        safety_vulns = len(self.results.get("safety", {}).get("vulnerabilities", []))
        total_issues += safety_vulns
        critical_issues += safety_vulns  # Dependencies are critical
        
        # Semgrep issues  
        semgrep_issues = len(self.results.get("semgrep", {}).get("issues", []))
        total_issues += semgrep_issues
        
        # Pip-audit vulnerabilities
        audit_vulns = len(self.results.get("pip_audit", {}).get("vulnerabilities", []))
        total_issues += audit_vulns
        critical_issues += audit_vulns  # Package vulnerabilities are critical
        
        report = {
            "summary": {
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "scan_status": "completed"
            },
            "tool_results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        if self.results.get("bandit", {}).get("issues"):
            recommendations.append("Review bandit security findings and fix high-severity issues")
        
        if self.results.get("safety", {}).get("vulnerabilities"):
            recommendations.append("Update vulnerable dependencies identified by safety")
        
        if self.results.get("semgrep", {}).get("issues"): 
            recommendations.append("Address semgrep static analysis security findings")
        
        if self.results.get("pip_audit", {}).get("vulnerabilities"):
            recommendations.append("Update packages with known vulnerabilities found by pip-audit")
        
        if not recommendations:
            recommendations.append("No security issues detected - continue monitoring")
        
        return recommendations
    
    def print_summary(self):
        """Print security scan summary."""
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("🔒 SECURITY SCAN SUMMARY")
        print("="*60)
        print(f"Total Issues: {summary['total_issues']}")
        print(f"Critical Issues: {summary['critical_issues']}")
        print(f"Scan Status: {summary['scan_status']}")
        
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print("="*60)


def main():
    """Main entry point for security scanning."""
    parser = argparse.ArgumentParser(description="Run security scans for refunc")
    parser.add_argument("--tool", choices=["bandit", "safety", "semgrep", "pip-audit", "all"], 
                       default="all", help="Security tool to run")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--output", help="Output file for report (JSON)")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    scanner = SecurityScanner(project_root)
    
    if args.tool == "all":
        results = scanner.run_all_scans()
    elif args.tool == "bandit":
        results = {"bandit": scanner.run_bandit()}
    elif args.tool == "safety":
        results = {"safety": scanner.run_safety()}
    elif args.tool == "semgrep":
        results = {"semgrep": scanner.run_semgrep()}
    elif args.tool == "pip-audit":
        results = {"pip_audit": scanner.run_pip_audit()}
    
    if args.report or args.output:
        report = scanner.generate_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Security report saved to: {args.output}")
        else:
            print(f"\n📄 Security Report:")
            print(json.dumps(report, indent=2))
    
    scanner.print_summary()
    
    # Exit with error code if critical issues found
    report = scanner.generate_report()
    if report["summary"]["critical_issues"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()