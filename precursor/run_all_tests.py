#!/usr/bin/env python3
"""
Run All Module Tests
====================

Master script to run all module tests in sequence.
Each test uses real data from: precursor/public/proteomics/BSA1.mzML
All results saved to: results/tests/

Author: Kundai Sachikonye
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_test(test_script, test_name):
    """Run a single test script."""
    print("\n" + "="*70)
    print(f"RUNNING: {test_name}")
    print("="*70)

    try:
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            print(f"\n✓ {test_name} PASSED")
            return True
        else:
            print(f"\n✗ {test_name} FAILED (exit code {result.returncode})")
            return False

    except Exception as e:
        print(f"\n✗ {test_name} FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DATABASE-FREE PROTEOMICS - COMPLETE TEST SUITE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test data: precursor/public/proteomics/BSA1.mzML")
    print("="*70)

    # Define all tests
    tests = [
        ("test_molecular_language.py", "Module 1: Molecular Language"),
        ("test_dictionary.py", "Module 2: S-Entropy Dictionary"),
        ("test_sequence_reconstruction.py", "Module 3: Sequence Reconstruction"),
        ("test_mmd_system_complete.py", "Module 4: Complete MMD System"),
    ]

    results = {}

    # Run each test
    for test_script, test_name in tests:
        test_path = Path(__file__).parent / test_script

        if not test_path.exists():
            print(f"\n⚠ Test script not found: {test_script}")
            results[test_name] = False
            continue

        results[test_name] = run_test(test_path, test_name)

    # Print final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n  🎉 ALL TESTS PASSED!")
        print(f"\n  Results saved to: results/tests/")
        print("     - molecular_language/")
        print("     - dictionary/")
        print("     - sequence/")
        print("     - mmd_system/")
    else:
        print(f"\n  ⚠ {total_count - passed_count} test(s) failed")

    print("="*70 + "\n")

    return passed_count == total_count


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
