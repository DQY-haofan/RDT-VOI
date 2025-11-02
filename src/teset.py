"""
并行执行专项诊断：定位pickle序列化问题

使用方法：
    python diagnose_parallel.py
"""

import sys
from pathlib import Path
import pickle
import traceback
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import load_scenario_config
from geometry import build_grid2d_geometry
from spatial_field import build_prior, sample_gmrf
from sensors import generate_sensor_pool
from method_wrappers import GreedyMIWrapper


def test_decision_config_pickle():
    """测试decision_config的序列化"""
    print("\n" + "="*70)
    print("  [1] TESTING DECISION_CONFIG PICKLE")
    print("="*70)

    cfg = load_scenario_config('A')

    print("\n[Before pickle]")
    for attr in ['L_TP_gbp', 'L_FP_gbp', 'L_FN_gbp', 'L_TN_gbp', 'tau_iri', 'tau_quantile']:
        val = getattr(cfg.decision, attr, "MISSING")
        print(f"  {attr}: {val} (type: {type(val).__name__})")

    # 尝试pickle
    try:
        pickled = pickle.dumps(cfg.decision)
        print(f"\n✅ Pickle succeeded ({len(pickled)} bytes)")

        # 尝试unpickle
        restored = pickle.loads(pickled)
        print(f"✅ Unpickle succeeded")

        print("\n[After unpickle]")
        issues = []
        for attr in ['L_TP_gbp', 'L_FP_gbp', 'L_FN_gbp', 'L_TN_gbp', 'tau_iri', 'tau_quantile']:
            val = getattr(restored, attr, "MISSING")
            print(f"  {attr}: {val} (type: {type(val).__name__})")

            # 检查是否变成了None
            original = getattr(cfg.decision, attr, "MISSING")
            if original is not None and val is None:
                issues.append(f"❌ {attr} became None after pickle!")
            elif original != val and attr != 'tau_iri':  # tau_iri可能初始就是None
                issues.append(f"⚠️  {attr} changed: {original} → {val}")

        if issues:
            print("\n❌ PICKLE ISSUES:")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("\n✅ decision_config pickle OK")
            return True

    except Exception as e:
        print(f"\n❌ Pickle failed: {e}")
        traceback.print_exc()
        return False


def test_fold_data_pickle():
    """测试完整fold_data的序列化"""
    print("\n" + "="*70)
    print("  [2] TESTING COMPLETE FOLD_DATA PICKLE")
    print("="*70)

    cfg = load_scenario_config('A')
    rng = cfg.get_rng()

    # 构建完整数据
    geom = build_grid2d_geometry(10, 10, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)
    x_true = sample_gmrf(Q_pr, mu_pr, rng)
    sensors = generate_sensor_pool(geom, cfg.sensors, rng)

    # 缓存tau
    tau = cfg.decision.get_threshold(mu_pr)
    cfg.decision.tau_iri = tau
    print(f"\n✅ Cached tau: {tau:.3f}")

    # 准备fold_data
    selection_method = GreedyMIWrapper(cfg)

    train_idx = rng.choice(geom.n, size=50, replace=False)
    test_idx = np.setdiff1d(np.arange(geom.n), train_idx)[:30]

    fold_data = {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'selection_method': selection_method,
        'k': 1,
        'Q_pr': Q_pr,
        'mu_pr': mu_pr,
        'x_true': x_true,
        'sensors': sensors,
        'decision_config': cfg.decision,
        'n_domain': geom.n,
        'coords': geom.coords,
        'adjacency_test': None,
        'rng_seed': rng.integers(0, 2**31),
        'enable_domain_scaling': True,
        'scenario': 'A',
        'morans_permutations': 999,
        'verbose': True
    }

    print("\n[Before pickle]")
    print(f"  decision_config.L_FP_gbp: {fold_data['decision_config'].L_FP_gbp}")
    print(f"  decision_config.tau_iri: {fold_data['decision_config'].tau_iri}")
    print(f"  sensors[0].cost: {fold_data['sensors'][0].cost}")

    # 尝试pickle
    try:
        print("\n[Pickling...]")
        pickled = pickle.dumps(fold_data)
        print(f"✅ Pickle succeeded ({len(pickled)} bytes)")

        print("\n[Unpickling...]")
        restored = pickle.loads(pickled)
        print(f"✅ Unpickle succeeded")

        print("\n[After unpickle]")
        issues = []

        # 检查decision_config
        dc = restored['decision_config']
        for attr in ['L_TP_gbp', 'L_FP_gbp', 'L_FN_gbp', 'L_TN_gbp', 'tau_iri']:
            val = getattr(dc, attr, "MISSING")
            original = getattr(fold_data['decision_config'], attr, "MISSING")

            print(f"  decision_config.{attr}: {val}")

            if original is not None and val is None:
                issues.append(f"❌ decision_config.{attr} became None!")

        # 检查sensors
        print(f"\n  sensors[0].cost: {restored['sensors'][0].cost}")
        if restored['sensors'][0].cost is None:
            issues.append(f"❌ sensors[0].cost became None!")

        # 检查selection_method
        print(f"  selection_method type: {type(restored['selection_method']).__name__}")

        if issues:
            print("\n❌ FOLD_DATA PICKLE ISSUES:")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("\n✅ fold_data pickle OK")
            return restored

    except Exception as e:
        print(f"\n❌ Pickle failed: {e}")
        traceback.print_exc()
        return None


def test_parallel_worker_execution(fold_data):
    """测试实际的worker函数执行"""
    print("\n" + "="*70)
    print("  [3] TESTING PARALLEL WORKER EXECUTION")
    print("="*70)

    from concurrent.futures import ProcessPoolExecutor
    from main import run_single_fold_worker

    print("\n[Submitting to process pool...]")

    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_single_fold_worker, fold_data)
            result = future.result(timeout=120)

        if result['success']:
            print("\n✅ Worker execution succeeded!")
            print(f"\n[Sample metrics]")
            for key in ['rmse', 'expected_loss_gbp', 'roi']:
                if key in result['metrics']:
                    print(f"  {key}: {result['metrics'][key]}")
            return True
        else:
            print(f"\n❌ Worker execution failed!")
            print(f"  Error: {result.get('error', 'unknown')}")

            if 'traceback' in result:
                print("\n[Full Traceback]")
                print(result['traceback'])

            return False

    except Exception as e:
        print(f"\n❌ Worker crashed: {e}")
        traceback.print_exc()
        return False


def test_tau_computation_in_worker():
    """专门测试worker中的tau计算"""
    print("\n" + "="*70)
    print("  [4] TESTING TAU COMPUTATION IN WORKER")
    print("="*70)

    cfg = load_scenario_config('A')
    rng = cfg.get_rng()

    geom = build_grid2d_geometry(10, 10, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)

    print("\n[Scenario 1: tau_iri NOT cached]")
    cfg.decision.tau_iri = None

    print(f"  decision_config.tau_iri: {cfg.decision.tau_iri}")
    print(f"  decision_config.tau_quantile: {cfg.decision.tau_quantile}")

    try:
        tau1 = cfg.decision.get_threshold(mu_pr)
        print(f"  ✅ Computed tau: {tau1:.3f}")
    except Exception as e:
        print(f"  ❌ get_threshold failed: {e}")
        return False

    print("\n[Scenario 2: tau_iri cached]")
    cfg.decision.tau_iri = tau1

    try:
        tau2 = cfg.decision.get_threshold(mu_pr)
        print(f"  ✅ Retrieved cached tau: {tau2:.3f}")

        if tau1 != tau2:
            print(f"  ⚠️  Tau mismatch: {tau1:.3f} vs {tau2:.3f}")
    except Exception as e:
        print(f"  ❌ get_threshold failed: {e}")
        return False

    print("\n[Scenario 3: After pickle]")
    pickled_cfg = pickle.loads(pickle.dumps(cfg.decision))

    print(f"  pickled_cfg.tau_iri: {pickled_cfg.tau_iri}")

    if pickled_cfg.tau_iri is None and cfg.decision.tau_iri is not None:
        print(f"  ❌ tau_iri lost after pickle!")
        return False
    else:
        print(f"  ✅ tau_iri preserved after pickle")

    return True


def test_expected_loss_with_none_check():
    """测试expected_loss函数中的None检查"""
    print("\n" + "="*70)
    print("  [5] TESTING EXPECTED_LOSS WITH NONE CHECKS")
    print("="*70)

    cfg = load_scenario_config('A')
    rng = cfg.get_rng()

    geom = build_grid2d_geometry(10, 10, h=cfg.geometry.h)
    Q_pr, mu_pr = build_prior(geom, cfg.prior)

    # 模拟后验数据
    n_test = 50
    mu_post = rng.normal(mu_pr.mean(), 0.5, n_test)
    sigma_post = rng.uniform(0.2, 0.5, n_test)

    print("\n[Test 1: With cached tau]")
    tau = cfg.decision.get_threshold(mu_pr)
    cfg.decision.tau_iri = tau

    from decision import expected_loss

    try:
        loss1 = expected_loss(mu_post, sigma_post, cfg.decision, tau=tau)
        print(f"  ✅ Loss computed: £{loss1:.2f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        traceback.print_exc()
        return False

    print("\n[Test 2: Without tau parameter]")
    try:
        loss2 = expected_loss(mu_post, sigma_post, cfg.decision)
        print(f"  ✅ Loss computed: £{loss2:.2f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        traceback.print_exc()
        return False

    print("\n[Test 3: After pickle with cached tau]")
    pickled_cfg = pickle.loads(pickle.dumps(cfg.decision))

    try:
        loss3 = expected_loss(mu_post, sigma_post, pickled_cfg)
        print(f"  ✅ Loss computed: £{loss3:.2f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        print(f"  pickled_cfg.tau_iri: {pickled_cfg.tau_iri}")
        print(f"  pickled_cfg.L_FP_gbp: {pickled_cfg.L_FP_gbp}")
        traceback.print_exc()
        return False

    return True


def main():
    """运行所有并行诊断测试"""
    print("\n" + "="*70)
    print("  PARALLEL EXECUTION DIAGNOSTICS")
    print("="*70)

    results = {}

    # Test 1: decision_config pickle
    results['decision_config_pickle'] = test_decision_config_pickle()

    # Test 2: fold_data pickle
    restored_fold_data = test_fold_data_pickle()
    results['fold_data_pickle'] = restored_fold_data is not None

    # Test 3: tau computation
    results['tau_computation'] = test_tau_computation_in_worker()

    # Test 4: expected_loss with None checks
    results['expected_loss'] = test_expected_loss_with_none_check()

    # Test 5: parallel worker (only if previous tests passed)
    if all(results.values()) and restored_fold_data:
        results['parallel_worker'] = test_parallel_worker_execution(restored_fold_data)
    else:
        print("\n⚠️  Skipping parallel worker test due to earlier failures")
        results['parallel_worker'] = False

    # Summary
    print("\n" + "="*70)
    print("  DIAGNOSTIC SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    if all(results.values()):
        print("\n🎉 All tests passed!")
        print("   The parallel execution should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Root cause identified:")

        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"\n  Failed tests: {', '.join(failed_tests)}")

        if not results['decision_config_pickle']:
            print("\n  💡 FIX: Check dataclass serialization in config.py")
            print("     Ensure all DecisionConfig attributes are picklable")

        if not results['fold_data_pickle']:
            print("\n  💡 FIX: Check fold_data preparation in main.py")
            print("     Some objects may not be picklable")

        if not results['tau_computation']:
            print("\n  💡 FIX: Check decision.py::DecisionConfig.get_threshold()")
            print("     Ensure tau_iri is properly cached and preserved")

        if not results['expected_loss']:
            print("\n  💡 FIX: Check decision.py::expected_loss()")
            print("     Add None checks for decision_config attributes")

        if not results['parallel_worker']:
            print("\n  💡 FIX: Check main.py::run_single_fold_worker()")
            print("     Add defensive None checks at the start of the function")


if __name__ == "__main__":
    main()