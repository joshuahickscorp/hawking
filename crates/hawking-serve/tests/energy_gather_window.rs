use hawking_serve::EnergyMode;
#[test]
fn single_slot_server_never_gathers() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        assert!(
            !mode.should_gather(1, 1),
            "{mode}: single-slot server must not delay the lone request"
        );
    }
}
#[test]
fn partial_batch_gathers_when_window_open() {
    assert!(!EnergyMode::Off.should_gather(1, 8), "off has no window");
    assert!(
        EnergyMode::Balanced.should_gather(1, 8),
        "balanced gathers a partial batch"
    );
    assert!(
        EnergyMode::Efficient.should_gather(1, 8),
        "efficient gathers a partial batch"
    );
    assert_eq!(EnergyMode::Off.gather_window_ms(), 0);
    assert_eq!(EnergyMode::Balanced.gather_window_ms(), 3);
    assert_eq!(EnergyMode::Efficient.gather_window_ms(), 8);
}
#[test]
fn full_batch_commits_immediately() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        assert!(
            !mode.should_gather(8, 8),
            "{mode}: a full batch must dispatch without waiting"
        );
        assert!(!mode.should_gather(9, 8), "{mode}: over-full never waits");
    }
}
#[test]
fn empty_queue_never_gathers() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        assert!(
            !mode.should_gather(0, 8),
            "{mode}: empty queue must not sleep"
        );
    }
}
#[test]
fn helper_matches_inline_loop_predicate() {
    for mode in [EnergyMode::Off, EnergyMode::Balanced, EnergyMode::Efficient] {
        for max_batch in [1usize, 2, 4, 8] {
            for ready in 0..=max_batch + 1 {
                let want =
                    ready > 0 && max_batch > 1 && ready < max_batch && mode.gather_window_ms() > 0;
                assert_eq!(mode.should_gather(ready, max_batch), want);
            }
        }
    }
}
