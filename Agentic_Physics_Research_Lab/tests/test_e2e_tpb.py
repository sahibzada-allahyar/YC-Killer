import subprocess, os

def test_tpb_suite():
    rc = subprocess.call(["python", "-m", "evaluation.run", "--suite", "evaluation/suites/tpb.yaml"])
    assert rc == 0
