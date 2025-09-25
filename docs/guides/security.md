# Security Scanning Guide

This guide covers the security scanning and vulnerability assessment system in refunc.

## Overview

The refunc security scanning suite provides comprehensive security analysis to ensure the library maintains high security standards. It includes:

- **Static security analysis** with bandit
- **Dependency vulnerability scanning** with safety and pip-audit
- **Code security patterns** with semgrep
- **CI/CD integration** for automated security monitoring

## Quick Start

### Running Security Scans

```bash
# Run comprehensive security scan
python scripts/security_scan.py

# Run specific security tool
python scripts/security_scan.py --tool bandit
python scripts/security_scan.py --tool safety
python scripts/security_scan.py --tool semgrep
python scripts/security_scan.py --tool pip-audit

# Generate detailed report
python scripts/security_scan.py --report --output security-report.json
```

### Individual Tool Usage

```bash
# Bandit - Python security linting
bandit -r refunc/ -f json

# Safety - Dependency vulnerability scanning
safety check --json

# Semgrep - Static analysis security scanning
semgrep --config .semgrep.yml refunc/

# Pip-audit - Package vulnerability scanning
pip-audit --format json
```

## Security Tools

### 1. Bandit - Python Security Linting

**Purpose**: Detects common security issues in Python code

**Configuration**: `.bandit`

```ini
[bandit]
level = medium
confidence = medium
exclude_dirs = tests,venv,.venv,build,dist
```

**Common Issues Detected**:
- Hardcoded passwords
- SQL injection vulnerabilities
- Command injection risks
- Insecure random number generation
- Dangerous function usage (eval, exec)

### 2. Safety - Dependency Vulnerability Scanning

**Purpose**: Scans Python dependencies for known security vulnerabilities

**Configuration**: `.safety-policy.json`

```json
{
  "security": {
    "ignore-vulnerabilities": [],
    "continue-on-vulnerability-error": false
  },
  "alert": {
    "ignore-cvss-severity-below": 0.0
  }
}
```

**Features**:
- CVE database lookup
- CVSS severity scoring
- Dependency tree analysis
- Policy-based filtering

### 3. Semgrep - Static Analysis Security Scanning

**Purpose**: Advanced static analysis for security patterns

**Configuration**: `.semgrep.yml`

```yaml
rules:
  - id: python-security-best-practices
    pattern-either:
      - pattern: eval(...)
      - pattern: exec(...)
      - pattern: os.system($X + ...)
    message: Potential security issue detected
    languages: [python]
    severity: WARNING
```

**Capabilities**:
- Custom security rules
- OWASP Top 10 patterns
- Path traversal detection
- Injection vulnerability detection

### 4. Pip-audit - Package Vulnerability Scanning

**Purpose**: Scans installed Python packages for known vulnerabilities

**Features**:
- PyPI security database
- OSV database integration
- Transitive dependency scanning
- SBOM (Software Bill of Materials) generation

## Security Configuration

### Pre-commit Hooks

Security scans are integrated into pre-commit hooks:

```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/PyCQA/bandit
  rev: '1.7.5'
  hooks:
  - id: bandit
    args: ['-c', '.bandit']
    exclude: ^tests/

- repo: https://github.com/gitguardian/ggshield
  rev: v1.25.0
  hooks:
  - id: ggshield
    language: python
    stages: [commit]
```

### CI/CD Integration

GitHub Actions workflow (`.github/workflows/security.yml`):

```yaml
- name: Run comprehensive security scan
  run: python scripts/security_scan.py --output security-report.json
  
- name: Fail on critical security issues
  run: |
    if [ -f security-report.json ]; then
      python -c "
      import json, sys
      with open('security-report.json', 'r') as f:
          report = json.load(f)
      if report['summary']['critical_issues'] > 0:
          sys.exit(1)
      "
    fi
```

## Security Best Practices

### Code Security

1. **Input Validation**: Always validate and sanitize inputs
2. **Avoid Dangerous Functions**: Don't use `eval()`, `exec()`, or `os.system()`
3. **Secure File Operations**: Use `pathlib` and validate file paths
4. **Secret Management**: Never hardcode secrets in source code

### Dependency Security

1. **Regular Updates**: Keep dependencies updated to latest secure versions
2. **Vulnerability Monitoring**: Use `safety` and `pip-audit` regularly
3. **Minimal Dependencies**: Only include necessary dependencies
4. **Dependency Pinning**: Pin dependency versions for reproducible builds

### Example Secure Code Patterns

```python
# ✅ Good: Secure file handling
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Securely load configuration file."""
    path = Path(config_path).resolve()
    
    # Validate path is within expected directory
    if not path.is_relative_to(Path.cwd()):
        raise SecurityError("Invalid config path")
    
    with path.open('r') as f:
        return json.load(f)

# ✅ Good: Parameterized queries
def get_user_data(user_id: int) -> dict:
    """Safely query user data."""
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()

# ❌ Bad: Command injection risk
def bad_example(filename: str):
    os.system(f"rm {filename}")  # Vulnerable to injection

# ✅ Good: Safe subprocess usage
def good_example(filename: str):
    subprocess.run(["rm", filename], check=True)
```

## Security Report Interpretation

### Report Structure

```json
{
  "summary": {
    "total_issues": 5,
    "critical_issues": 1,
    "scan_status": "completed"
  },
  "tool_results": {
    "bandit": {"status": "issues_found", "issues": [...]},
    "safety": {"status": "vulnerabilities_found", "vulnerabilities": [...]},
    "semgrep": {"status": "success", "issues": []},
    "pip_audit": {"status": "success", "vulnerabilities": []}
  },
  "recommendations": [
    "Update vulnerable dependencies identified by safety",
    "Review bandit security findings and fix high-severity issues"
  ]
}
```

### Severity Levels

- **Critical**: Immediate security risk, requires urgent fix
- **High**: Significant security concern, should be fixed soon
- **Medium**: Moderate security issue, fix in next release
- **Low**: Minor security improvement, fix when convenient

### Common False Positives

1. **Test files**: Security tools may flag test code - use excludes
2. **Development tools**: Dev dependencies may have vulnerabilities not affecting production
3. **Documentation examples**: Example code may trigger security warnings

## Handling Security Issues

### Vulnerability Response Process

1. **Assessment**: Evaluate the severity and impact
2. **Patching**: Apply security fixes or updates
3. **Testing**: Verify the fix doesn't break functionality
4. **Documentation**: Update security documentation if needed

### Dependency Vulnerabilities

```bash
# Update vulnerable dependency
pip install --upgrade vulnerable-package

# Check if update fixes the issue
safety check

# Update requirements files
pip freeze > requirements/base.txt
```

### Code Security Issues

```python
# Before: Vulnerable to injection
def execute_command(cmd: str):
    os.system(cmd)

# After: Safe subprocess usage
def execute_command(cmd: List[str]):
    return subprocess.run(cmd, check=True, capture_output=True)
```

## Advanced Security Features

### Custom Security Rules

Create custom semgrep rules for project-specific security patterns:

```yaml
# .semgrep-custom.yml
rules:
  - id: refunc-specific-patterns
    pattern: |
      def $FUNC(...):
        ...
        eval($VAR)
        ...
    message: Avoid eval() in refunc functions
    languages: [python]
    severity: ERROR
```

### Security Policy Integration

```python
# security_policy.py
class SecurityPolicy:
    """Enforce security policies in refunc."""
    
    @staticmethod
    def validate_file_path(path: str) -> bool:
        """Validate file path for security."""
        path_obj = Path(path).resolve()
        
        # Prevent path traversal
        if ".." in path:
            return False
        
        # Ensure within allowed directories
        allowed_dirs = [Path.cwd(), Path.home() / ".refunc"]
        return any(path_obj.is_relative_to(d) for d in allowed_dirs)
```

## Monitoring and Alerting

### GitHub Security Advisories

Configure GitHub Dependabot for automatic security updates:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
    open-pull-requests-limit: 10
```

### Security Notifications

Set up notifications for security issues:

1. **GitHub notifications**: Enable security alerts
2. **CI/CD alerts**: Configure failure notifications
3. **Email alerts**: Use safety's email notifications
4. **Slack integration**: Webhook notifications for critical issues

## Related Documentation

- [Contributing Guidelines](../developer/contributing.md) - Security requirements for contributors
- [Installation Guide](installation.md) - Secure installation practices
- [Configuration Guide](../api/config.md) - Secure configuration management