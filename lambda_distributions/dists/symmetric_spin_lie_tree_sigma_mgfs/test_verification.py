from lambda_distributions.dists.symmetric_spin_lie_tree_sigma_mgfs.verification import run_suite


def test_full_verification_suite():
    report = run_suite()
    assert report["passed"]
    assert report["exact checks"] == 2036
    assert report["symmetric groups"]["character-power checks"] == 1152
    assert report["Schur cover"]["group order"] == 48
    assert report["Foulkes"]["character checks"] == 744
    assert report["free Lie"]["cases"][-1]["second moment formula"] == 2
    assert report["rooted trees"]["iterated cases"][-1]["pair moment formula"] == 3

