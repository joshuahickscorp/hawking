//! hide-browser: the browser evidence and visual verification model.
//!
//! HIDE treats "did the change work in the browser?" as a first-class,
//! evidence-backed question (Bible Book VIII sec 27). This crate defines the
//! shapes that answer it and the deterministic machinery that grades them:
//!
//! - [`evidence`]: a [`BrowserStep`] evidence record (url, navigation cause, DOM
//!   snapshot, accessibility tree, screenshot ref, console + network events,
//!   the selected element, the action, the resulting state, timing) and a
//!   [`BrowserSession`] -- an ordered, replayable list of steps.
//! - [`driver`]: a [`BrowserDriver`] trait (navigate/click/fill + screenshot/
//!   dom/accessibility/console/network) and a [`ReplayDriver`] that plays a
//!   recorded session back deterministically.
//! - [`acceptance`]: a [`VisualAcceptance`] oracle with a deterministic
//!   functional evaluator that reads a recorded [`ResultingState`] and returns a
//!   typed [`Verdict`], plus a deterministic a11y evaluator over the recorded
//!   accessibility tree.
//! - [`design_mode`]: [`DesignAnnotation`] mapping a selected element to its DOM
//!   node, source symbol, CSS rule, layout box, and accessibility node.
//!
//! # Model-free
//!
//! This crate never opens a network connection, never drives a real browser,
//! and runs no model (RIP doctrine). Everything is proven with deterministic
//! tests over recorded fixtures. Heavy artifacts (screenshots, raw HTML,
//! network bodies) are carried by content-addressed [`ArtifactRef`], never
//! inlined.
//!
//! The legs that inherently need a real renderer or a model are marked
//! `DEFERRED_MODEL_REQUIRED` at their definitions and are NOT implemented or
//! claimed here:
//!
//! - a live chromium/CDP backend that produces evidence from a real page
//!   (a future implementor of [`BrowserDriver`]; see [`driver`]);
//! - pixel/appearance grading of a screenshot against a target within
//!   tolerances, and judging `semantic_requirements`
//!   (see [`VisualAcceptance::evaluate_visual`]).
//!
//! Wire shapes here are HIDE-native. Where a concept is borrowed from an open
//! spec it is noted at the definition as spec-derived (for example the ARIA
//! interactive-role set in [`a11y`]); no proprietary source is copied.
//!
//! ```
//! use hide_browser::{ElementSelector, StateCheck, VisualAcceptance, ResultingState};
//!
//! let acc = VisualAcceptance {
//!     id: "va".into(),
//!     target: None,
//!     responsive_states: vec![],
//!     semantic_requirements: vec![],
//!     functional_interactions: vec![hide_browser::FunctionalRequirement {
//!         id: "cart".into(),
//!         description: "cart shows one item".into(),
//!         check: StateCheck::SignalEquals { key: "cart.count".into(), value: "1".into() },
//!     }],
//!     a11y_requirements: vec![],
//!     tolerances: Default::default(),
//!     before_after: None,
//! };
//! let good = ResultingState::at("https://x/").signal("cart.count", "1");
//! assert!(acc.evaluate_functional(&good).is_pass());
//! let _ = ElementSelector::css("#add");
//! ```

pub mod a11y;
pub mod acceptance;
pub mod design_mode;
pub mod dom;
pub mod driver;
pub mod error;
pub mod evidence;
pub mod ids;

pub use a11y::{is_interactive_role, AccessibilityNode, AccessibilityTree, INTERACTIVE_ROLES};
pub use acceptance::{
    A11yCheck, A11yRequirement, BeforeAfter, FailureKind, FailureReason, FunctionalRequirement,
    ResponsiveTarget, SemanticRequirement, StateCheck, Tolerances, Verdict, VisualAcceptance,
};
pub use design_mode::{annotate, DesignAnnotation};
pub use dom::{BoxModel, CssRule, DomNode, DomSnapshot, EdgeSizes, SelectStrategy, SourceSymbol};
pub use driver::{BrowserDriver, ReplayDriver};
pub use error::{BrowserError, Result};
pub use evidence::{
    BrowserAction, BrowserSession, BrowserStep, ConsoleEvent, ConsoleLevel, ElementSelector,
    NavigationCause, NetworkEvent, ResponsiveState, ResultingState, Viewport,
};
pub use ids::{AccessibilityNodeId, ArtifactRef, BrowserSessionId, DomNodeId};

/// Generate the JSON Schema for a type as a [`serde_json::Value`]. The schema is
/// derived from the Rust definition, never hand-maintained, so it cannot drift
/// from the shape the code serializes.
pub fn json_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let root = schemars::gen::SchemaGenerator::default().into_root_schema_for::<T>();
    serde_json::to_value(root).expect("a schemars RootSchema always serializes to JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // -- fixture: a recorded "add to cart" session -------------------------

    const PRODUCT_URL: &str = "https://shop.test/product/42";
    const CHECKOUT_URL: &str = "https://shop.test/checkout";

    fn page_dom() -> DomSnapshot {
        let mut btn = DomNode::leaf("n-btn", "button");
        btn.attributes.insert("id".into(), "add-to-cart".into());
        btn.attributes.insert("data-testid".into(), "add-cart".into());
        btn.text = Some("Add to cart".into());
        btn.box_model = Some(BoxModel {
            x: 20.0,
            y: 100.0,
            width: 140.0,
            height: 44.0,
            ..Default::default()
        });
        btn.a11y_node = Some(AccessibilityNodeId::from("ax-btn"));
        btn.source_symbol = Some(SourceSymbol {
            file: "app/ProductPage.tsx".into(),
            symbol: "AddToCartButton".into(),
            line: Some(42),
            column: None,
        });
        let mut decls = BTreeMap::new();
        decls.insert("background".into(), "#070707".into());
        btn.css_rules = vec![CssRule {
            selector: "#add-to-cart".into(),
            source: Some("app/product.css".into()),
            declarations: decls,
        }];

        let mut coupon = DomNode::leaf("n-coupon", "input");
        coupon.attributes.insert("id".into(), "coupon".into());
        coupon.a11y_node = Some(AccessibilityNodeId::from("ax-coupon"));

        let mut root = DomNode::leaf("n-root", "main");
        root.children.push(btn);
        root.children.push(coupon);
        DomSnapshot::new(root)
    }

    fn page_a11y(button_named: bool) -> AccessibilityTree {
        let button = if button_named {
            AccessibilityNode::new("ax-btn", "button").with_name("Add to cart")
        } else {
            AccessibilityNode::new("ax-btn", "button")
        };
        AccessibilityTree::new(
            AccessibilityNode::new("ax-root", "document")
                .with_child(button)
                .with_child(AccessibilityNode::new("ax-coupon", "textbox").with_name("Coupon code")),
        )
    }

    fn network_get() -> NetworkEvent {
        NetworkEvent {
            request_id: "req-1".into(),
            method: "GET".into(),
            url: PRODUCT_URL.into(),
            resource_type: Some("document".into()),
            status: Some(200),
            request_ref: None,
            response_ref: Some(ArtifactRef::content_addressed(
                b"<html>product</html>",
                Some("text/html"),
            )),
            timing_ms: Some(80),
        }
    }

    fn session() -> BrowserSession {
        let step0 = BrowserStep {
            index: 0,
            url: PRODUCT_URL.into(),
            navigation_cause: NavigationCause::UserNavigate,
            action: BrowserAction::Navigate,
            selected_element: None,
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-0", Some("image/png"))),
            console_events: vec![],
            network_events: vec![network_get()],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(PRODUCT_URL)
                    .signal("cart.count", "0")
                    .present("#add-to-cart")
                    .present("#coupon")
                    .text("Product 42")
            },
            timing_ms: 120,
        };
        let step1 = BrowserStep {
            index: 1,
            url: PRODUCT_URL.into(),
            navigation_cause: NavigationCause::None,
            action: BrowserAction::Click,
            selected_element: Some(ElementSelector::css("#add-to-cart").with_node("n-btn")),
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-1", Some("image/png"))),
            console_events: vec![],
            network_events: vec![],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(PRODUCT_URL)
                    .signal("cart.count", "1")
                    .present("#add-to-cart")
                    .present("#coupon")
                    .present("#cart-toast")
                    .text("Added to cart")
            },
            timing_ms: 40,
        };
        let step2 = BrowserStep {
            index: 2,
            url: PRODUCT_URL.into(),
            navigation_cause: NavigationCause::None,
            action: BrowserAction::Fill {
                value: "SAVE10".into(),
            },
            selected_element: Some(ElementSelector::css("#coupon").with_node("n-coupon")),
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-2", Some("image/png"))),
            console_events: vec![],
            network_events: vec![],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(PRODUCT_URL)
                    .signal("cart.count", "1")
                    .signal("coupon", "SAVE10")
            },
            timing_ms: 30,
        };
        let step3 = BrowserStep {
            index: 3,
            url: CHECKOUT_URL.into(),
            navigation_cause: NavigationCause::FormSubmit,
            action: BrowserAction::Navigate,
            selected_element: None,
            dom_snapshot: page_dom(),
            accessibility_tree: page_a11y(true),
            screenshot_ref: Some(ArtifactRef::content_addressed(b"png-step-3", Some("image/png"))),
            console_events: vec![],
            network_events: vec![],
            resulting_state: ResultingState {
                http_status: Some(200),
                ..ResultingState::at(CHECKOUT_URL)
            },
            timing_ms: 200,
        };
        BrowserSession::new("bs-cart", vec![step0, step1, step2, step3])
    }

    // -- test 1: replay captures each step in order ------------------------

    #[test]
    fn replay_plays_recorded_session_step_by_step_in_order() {
        let recorded = session();
        // Round-trip through JSON to prove replay works from a serialized trace.
        let wire = serde_json::to_string(&recorded).unwrap();
        let restored: BrowserSession = serde_json::from_str(&wire).unwrap();
        assert_eq!(restored, recorded);
        assert!(restored.indices_are_sequential());

        let mut d = ReplayDriver::new(restored);
        assert_eq!(d.len(), 4);

        let mut played = Vec::new();
        played.push(d.navigate(PRODUCT_URL).unwrap());
        // observers read the current (last-played) step's evidence
        assert_eq!(
            d.screenshot().unwrap(),
            recorded.steps[0].screenshot_ref.clone().unwrap()
        );
        assert_eq!(d.dom().unwrap(), recorded.steps[0].dom_snapshot);
        assert_eq!(d.accessibility().unwrap(), recorded.steps[0].accessibility_tree);
        assert_eq!(d.network().unwrap(), recorded.steps[0].network_events);
        assert!(d.console().unwrap().is_empty());

        played.push(
            d.click(&ElementSelector::css("#add-to-cart"))
                .unwrap(),
        );
        played.push(d.fill(&ElementSelector::css("#coupon"), "SAVE10").unwrap());
        played.push(d.navigate(CHECKOUT_URL).unwrap());

        assert_eq!(played, recorded.steps, "each step replayed in recorded order");
        assert!(d.is_exhausted());

        // One more call runs off the end.
        assert_eq!(
            d.navigate("anywhere"),
            Err(BrowserError::ReplayExhausted { requested: "navigate" })
        );
    }

    #[test]
    fn replay_is_a_strict_contract_and_reports_mismatch() {
        let mut d = ReplayDriver::new(session());
        // The first recorded step is a navigate; asking to click instead is a
        // typed mismatch that names the recorded action and index.
        let err = d.click(&ElementSelector::css("#add-to-cart")).unwrap_err();
        match err {
            BrowserError::ReplayMismatch {
                index,
                requested,
                expected,
                ..
            } => {
                assert_eq!(index, 0);
                assert_eq!(requested, "click");
                assert_eq!(expected, "navigate");
            }
            other => panic!("expected a replay mismatch, got {other:?}"),
        }
        // A mismatch does not advance the cursor: the correct call still works.
        assert!(d.navigate(PRODUCT_URL).is_ok());
    }

    #[test]
    fn observers_error_before_any_step_is_played() {
        let d = ReplayDriver::new(session());
        assert_eq!(d.dom(), Err(BrowserError::NoCurrentStep));
        assert_eq!(d.screenshot(), Err(BrowserError::NoCurrentStep));
    }

    // -- test 2: functional oracle passes/fails with typed reasons ---------

    fn acceptance() -> VisualAcceptance {
        VisualAcceptance {
            id: "va-add-to-cart".into(),
            target: Some(ArtifactRef::new("design/add-to-cart.png")),
            responsive_states: vec![ResponsiveTarget {
                name: "mobile".into(),
                viewport: Viewport {
                    width: 375,
                    height: 812,
                },
                screenshot_ref: Some(ArtifactRef::new("design/add-to-cart.mobile.png")),
            }],
            semantic_requirements: vec![SemanticRequirement {
                id: "cta-prominent".into(),
                description: "the add-to-cart CTA is the visual focus".into(),
            }],
            functional_interactions: vec![
                FunctionalRequirement {
                    id: "cart-incremented".into(),
                    description: "cart shows one item after adding".into(),
                    check: StateCheck::SignalEquals {
                        key: "cart.count".into(),
                        value: "1".into(),
                    },
                },
                FunctionalRequirement {
                    id: "toast-shown".into(),
                    description: "a confirmation toast appears".into(),
                    check: StateCheck::ElementPresent {
                        selector: "#cart-toast".into(),
                    },
                },
                FunctionalRequirement {
                    id: "no-console-errors".into(),
                    description: "no console errors during the interaction".into(),
                    check: StateCheck::NoConsoleErrors,
                },
                FunctionalRequirement {
                    id: "stays-on-product".into(),
                    description: "still on the product page".into(),
                    check: StateCheck::UrlContains {
                        fragment: "/product/42".into(),
                    },
                },
            ],
            a11y_requirements: vec![
                A11yRequirement {
                    id: "button-has-name".into(),
                    description: "the add button has an accessible name".into(),
                    check: A11yCheck::RoleNamed {
                        role: "button".into(),
                    },
                },
                A11yRequirement {
                    id: "no-unnamed-controls".into(),
                    description: "every interactive control is named".into(),
                    check: A11yCheck::NoUnnamedInteractive,
                },
            ],
            tolerances: Tolerances::default(),
            before_after: Some(BeforeAfter {
                before: Some(ArtifactRef::new("art/before.png")),
                after: Some(ArtifactRef::new("art/after.png")),
            }),
        }
    }

    #[test]
    fn functional_oracle_passes_on_the_good_recorded_state() {
        let s = session();
        let good = &s.steps[1].resulting_state; // after the click
        let verdict = acceptance().evaluate_functional(good);
        assert!(verdict.is_pass(), "good state passes: {verdict:?}");
    }

    #[test]
    fn functional_oracle_fails_on_a_bad_state_with_typed_reasons() {
        let s = session();
        // Corrupt the post-click state: cart did not increment, toast missing,
        // and two console errors fired.
        let mut bad = s.steps[1].resulting_state.clone();
        bad.signals.insert("cart.count".into(), "0".into());
        bad.present_selectors.remove("#cart-toast");
        bad.console_error_count = 2;

        let verdict = acceptance().evaluate_functional(&bad);
        assert!(!verdict.is_pass());
        let reasons = verdict.reasons();
        assert_eq!(reasons.len(), 3, "three requirements failed: {reasons:?}");

        // Reasons are typed, not stringly. Find each expected failure kind.
        assert!(reasons.iter().any(|r| matches!(
            &r.kind,
            FailureKind::SignalMismatch { key, expected, actual }
                if key == "cart.count" && expected == "1" && actual == "0"
        )));
        assert!(reasons.iter().any(|r| matches!(
            &r.kind,
            FailureKind::ElementMissing { selector } if selector == "#cart-toast"
        )));
        assert!(reasons.iter().any(|r| matches!(
            &r.kind,
            FailureKind::ConsoleErrors { count, allowed } if *count == 2 && *allowed == 0
        )));
        // The typed reason carries its requirement id for traceability.
        assert!(reasons.iter().all(|r| !r.requirement_id.is_empty()));
    }

    #[test]
    fn a11y_oracle_passes_on_named_controls_and_fails_when_unnamed() {
        let acc = acceptance();
        assert!(acc.evaluate_a11y(&page_a11y(true)).is_pass());

        let verdict = acc.evaluate_a11y(&page_a11y(false));
        assert!(!verdict.is_pass());
        // Both the "button named" and "no unnamed interactive" checks fail.
        assert!(verdict
            .reasons()
            .iter()
            .any(|r| matches!(&r.kind, FailureKind::A11yUnnamed { role } if role == "button")));
    }

    // -- test 3: annotation maps a selection to a DOM node -----------------

    #[test]
    fn annotation_maps_a_selection_to_its_dom_node() {
        let dom = session().steps[0].dom_snapshot.clone();

        // With a pre-resolved node id.
        let sel = ElementSelector::css("#add-to-cart").with_node("n-btn");
        let ann = annotate(&sel, &dom).unwrap();
        assert_eq!(ann.dom_node, DomNodeId::from("n-btn"));
        assert_eq!(ann.a11y_node, Some(AccessibilityNodeId::from("ax-btn")));
        assert_eq!(
            ann.source_symbol.as_ref().map(|s| s.symbol.as_str()),
            Some("AddToCartButton")
        );
        assert_eq!(
            ann.css_rule.as_ref().map(|c| c.selector.as_str()),
            Some("#add-to-cart")
        );
        assert!(ann.box_model.width > 0.0);

        // Without a pre-resolved id: resolve structurally from the selector.
        let sel2 = ElementSelector::test_id("add-cart");
        let ann2 = annotate(&sel2, &dom).unwrap();
        assert_eq!(ann2.dom_node, DomNodeId::from("n-btn"));

        // An unresolvable selection is a typed error.
        let ghost = ElementSelector::css("#ghost");
        assert!(matches!(
            annotate(&ghost, &dom),
            Err(BrowserError::UnresolvedSelection { .. })
        ));
    }

    // -- test 4: heavy artifacts referenced by id, never inlined -----------

    #[test]
    fn screenshots_and_network_bodies_are_referenced_not_inlined() {
        let s = session();
        let step = &s.steps[0];

        // Screenshot is a content-addressed reference.
        let sref = step.screenshot_ref.as_ref().unwrap();
        assert!(sref.is_content_addressed());

        // Network response body is referenced, not inlined.
        let net = &step.network_events[0];
        assert!(net.response_ref.as_ref().unwrap().is_content_addressed());

        // The serialized step carries only the reference ids; the original
        // bytes (b"png-step-0", the HTML body) never appear on the wire.
        let json = serde_json::to_string(step).unwrap();
        assert!(json.contains(&sref.id), "the screenshot ref id is on the wire");
        assert!(!json.contains("png-step-0"), "screenshot bytes are not inlined");
        assert!(!json.contains("<html>product</html>"), "response body not inlined");
    }

    // -- serde + schema -----------------------------------------------------

    #[test]
    fn top_types_round_trip_through_serde_json() {
        let s = session();
        let back: BrowserSession =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);

        let acc = acceptance();
        let back_acc: VisualAcceptance =
            serde_json::from_str(&serde_json::to_string(&acc).unwrap()).unwrap();
        assert_eq!(back_acc, acc);

        let ann = annotate(
            &ElementSelector::css("#add-to-cart").with_node("n-btn"),
            &s.steps[0].dom_snapshot,
        )
        .unwrap();
        let back_ann: DesignAnnotation =
            serde_json::from_str(&serde_json::to_string(&ann).unwrap()).unwrap();
        assert_eq!(back_ann, ann);

        let verdict = acc.evaluate_functional(&s.steps[1].resulting_state);
        let back_v: Verdict =
            serde_json::from_str(&serde_json::to_string(&verdict).unwrap()).unwrap();
        assert_eq!(back_v, verdict);
    }

    #[test]
    fn json_schema_generates_for_the_core_types() {
        for schema in [
            json_schema::<BrowserStep>(),
            json_schema::<BrowserSession>(),
            json_schema::<VisualAcceptance>(),
            json_schema::<DesignAnnotation>(),
            json_schema::<Verdict>(),
        ] {
            assert!(schema.is_object(), "each schema is a JSON object");
            assert!(
                schema.get("$schema").is_some() || schema.get("title").is_some(),
                "a schemars root schema carries a $schema or title marker"
            );
        }
    }

    #[test]
    fn browser_action_serializes_internally_tagged() {
        let value = serde_json::to_value(BrowserAction::Fill {
            value: "hi".into(),
        })
        .unwrap();
        assert_eq!(value.get("kind").unwrap(), "fill");
        assert_eq!(value.get("value").unwrap(), "hi");
    }
}
