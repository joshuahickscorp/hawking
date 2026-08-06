fn e3_default(user_draft_on: bool, explicit: Option<bool>) -> bool {
    explicit.unwrap_or(!user_draft_on)
}
#[test]
fn explicit_override_always_wins() {
    assert!(
        e3_default(true, Some(true)),
        "explicit=1 forces E3 ON even under user-draft"
    );
    assert!(e3_default(false, Some(true)), "explicit=1 forces E3 ON");
    assert!(!e3_default(false, Some(false)), "explicit=0 forces E3 OFF");
    assert!(
        !e3_default(true, Some(false)),
        "explicit=0 forces E3 OFF even without user-draft"
    );
}
#[test]
fn unset_default_is_off_under_user_draft_on_otherwise() {
    assert!(
        e3_default(false, None),
        "unset + no user-draft => E3 ON (+9.6%)"
    );
    assert!(
        !e3_default(true, None),
        "unset + user-draft ON => E3 OFF (draft stays lossless)"
    );
}
