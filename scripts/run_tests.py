#!/usr/bin/env python3
"""
Comprehensive test runner for the ML optimization framework.

This script provides a unified interface for running different types of tests
including unit tests, integration tests, and framework validation.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger, setup_development_logging

# Setup logging
setup_development_logging()
logger = get_logger(__name__)


class TestRunner:
    """
    Comprehensive test runner for the ML optimization framework.
    
    Provides different test execution modes and comprehensive reporting.
    """
    
    def __init__(self):
        """Initialize the test runner."""
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """
        Run a command and capture results.
        
        Args:
            command: Command to run as list of strings
            description: Description of the command
            
        Returns:
            Dictionary with command results
        """
        logger.info(f"🔄 {description}...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.success(f"✅ {description} completed successfully ({execution_time:.2f}s)")
                return {
                    'success': True,
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'execution_time': execution_time
                }
            else:
                logger.error(f"❌ {description} failed ({execution_time:.2f}s)")
                logger.error(f"Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                return {
                    'success': False,
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'execution_time': execution_time
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {str(e)}")
            return {
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': time.time() - start_time
            }
    
    def run_unit_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run unit tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-m", "unit and not slow",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=70",
            "--tb=short"
        ]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Unit tests")
    
    def run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run integration tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-m", "integration",
            "--tb=short",
            "--timeout=300"
        ]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Integration tests")
    
    def run_smoke_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run smoke tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-m", "smoke",
            "--tb=short"
        ]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Smoke tests")
    
    def run_slow_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run slow tests."""
        command = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-m", "slow",
            "--tb=short",
            "--timeout=600"
        ]
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command, "Slow tests")
    
    def run_code_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks."""
        checks = {}
        
        # Black formatting check
        checks['black'] = self.run_command(
            [sys.executable, "-m", "black", "--check", "--diff", "src/", "tests/"],
            "Black formatting check"
        )
        
        # isort import sorting check
        checks['isort'] = self.run_command(
            [sys.executable, "-m", "isort", "--check-only", "--diff", "src/", "tests/"],
            "isort import sorting check"
        )
        
        # Flake8 linting
        checks['flake8'] = self.run_command(
            [sys.executable, "-m", "flake8", "src/", "tests/", 
             "--max-line-length=88", "--extend-ignore=E203,W503"],
            "Flake8 linting"
        )
        
        # MyPy type checking
        checks['mypy'] = self.run_command(
            [sys.executable, "-m", "mypy", "src/", 
             "--ignore-missing-imports", "--no-strict-optional"],
            "MyPy type checking"
        )
        
        return checks
    
    def run_framework_validation(self) -> Dict[str, Any]:
        """Run framework validation."""
        return self.run_command(
            [sys.executable, "scripts/validate_framework.py"],
            "Framework validation"
        )
    
    def run_all_tests(self, include_slow: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all tests.
        
        Args:
            include_slow: Whether to include slow tests
            verbose: Whether to run tests in verbose mode
            
        Returns:
            Dictionary with all test results
        """
        logger.info("🚀 Starting comprehensive test suite...")
        start_time = time.time()
        
        results = {}
        
        # Code quality checks
        logger.info("📋 Running code quality checks...")
        results['code_quality'] = self.run_code_quality_checks()
        
        # Unit tests
        results['unit_tests'] = self.run_unit_tests(verbose)
        
        # Integration tests
        results['integration_tests'] = self.run_integration_tests(verbose)
        
        # Smoke tests
        results['smoke_tests'] = self.run_smoke_tests(verbose)
        
        # Framework validation
        results['framework_validation'] = self.run_framework_validation()
        
        # Slow tests (optional)
        if include_slow:
            results['slow_tests'] = self.run_slow_tests(verbose)
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self.generate_summary(results, total_time)
        results['summary'] = summary
        
        return results
    
    def generate_summary(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """
        Generate test summary.
        
        Args:
            results: Test results dictionary
            total_time: Total execution time
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_time': total_time,
            'passed': 0,
            'failed': 0,
            'categories': {}
        }
        
        for category, result in results.items():
            if category == 'summary':
                continue
                
            if isinstance(result, dict):
                if 'success' in result:
                    # Single test result
                    if result['success']:
                        summary['passed'] += 1
                    else:
                        summary['failed'] += 1
                    summary['categories'][category] = result['success']
                else:
                    # Multiple test results (like code quality)
                    category_passed = 0
                    category_total = 0
                    for sub_test, sub_result in result.items():
                        category_total += 1
                        if sub_result.get('success', False):
                            category_passed += 1
                    
                    summary['categories'][category] = {
                        'passed': category_passed,
                        'total': category_total,
                        'success': category_passed == category_total
                    }
                    
                    if category_passed == category_total:
                        summary['passed'] += 1
                    else:
                        summary['failed'] += 1
        
        summary['success_rate'] = summary['passed'] / (summary['passed'] + summary['failed']) if (summary['passed'] + summary['failed']) > 0 else 0
        
        return summary
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print test summary."""
        summary = results.get('summary', {})
        
        logger.info("=" * 60)
        logger.info("🎯 TEST SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"⏱️ Total time: {summary.get('total_time', 0):.2f}s")
        logger.info(f"✅ Passed: {summary.get('passed', 0)}")
        logger.info(f"❌ Failed: {summary.get('failed', 0)}")
        logger.info(f"📊 Success rate: {summary.get('success_rate', 0):.1%}")
        
        logger.info("\n📋 Category Results:")
        for category, result in summary.get('categories', {}).items():
            if isinstance(result, bool):
                status = "✅" if result else "❌"
                logger.info(f"   {status} {category}")
            elif isinstance(result, dict):
                status = "✅" if result.get('success', False) else "❌"
                passed = result.get('passed', 0)
                total = result.get('total', 0)
                logger.info(f"   {status} {category}: {passed}/{total}")
        
        if summary.get('success_rate', 0) == 1.0:
            logger.success("\n🎉 All tests passed! Framework is ready for use.")
        else:
            logger.warning(f"\n⚠️ Some tests failed. Please review the results above.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for ML optimization framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests (excluding slow tests)
  python scripts/run_tests.py --all
  
  # Run only unit tests
  python scripts/run_tests.py --unit
  
  # Run code quality checks
  python scripts/run_tests.py --quality
  
  # Run all tests including slow tests
  python scripts/run_tests.py --all --include-slow
  
  # Run framework validation
  python scripts/run_tests.py --validate
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--smoke', action='store_true', help='Run smoke tests')
    parser.add_argument('--slow', action='store_true', help='Run slow tests')
    parser.add_argument('--quality', action='store_true', help='Run code quality checks')
    parser.add_argument('--validate', action='store_true', help='Run framework validation')
    parser.add_argument('--include-slow', action='store_true', help='Include slow tests in --all')
    parser.add_argument('--quiet', action='store_true', help='Run tests in quiet mode')
    
    args = parser.parse_args()
    
    # If no specific test type is specified, run all
    if not any([args.unit, args.integration, args.smoke, args.slow, args.quality, args.validate]):
        args.all = True
    
    runner = TestRunner()
    verbose = not args.quiet
    
    try:
        if args.all:
            results = runner.run_all_tests(include_slow=args.include_slow, verbose=verbose)
        else:
            results = {}
            
            if args.quality:
                results['code_quality'] = runner.run_code_quality_checks()
            
            if args.unit:
                results['unit_tests'] = runner.run_unit_tests(verbose)
            
            if args.integration:
                results['integration_tests'] = runner.run_integration_tests(verbose)
            
            if args.smoke:
                results['smoke_tests'] = runner.run_smoke_tests(verbose)
            
            if args.slow:
                results['slow_tests'] = runner.run_slow_tests(verbose)
            
            if args.validate:
                results['framework_validation'] = runner.run_framework_validation()
            
            # Generate summary for partial runs
            if results:
                total_time = sum(
                    result.get('execution_time', 0) 
                    for result in results.values() 
                    if isinstance(result, dict) and 'execution_time' in result
                )
                results['summary'] = runner.generate_summary(results, total_time)
        
        # Print summary
        if results:
            runner.print_summary(results)
            
            # Exit with appropriate code
            summary = results.get('summary', {})
            if summary.get('success_rate', 0) == 1.0:
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            logger.warning("No tests were run.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test runner failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
