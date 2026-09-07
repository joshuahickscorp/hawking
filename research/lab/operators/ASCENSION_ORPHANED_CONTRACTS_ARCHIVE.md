# Ascension orphaned contract archive

These eight operator modules were superseded planning/configuration surfaces. A
repository-wide tracked-text audit found no caller, test, receipt, or manifest
reference to any of them. They had no execution path and therefore did not
constitute current Ascension capability.

| Removed surface | Durable signal retained by the canonical owner |
| --- | --- |
| `ascension_execution_plan.py` | lifecycle state and the V3 execution sequence |
| `ascension_family_workflow.py` | family rules and measured campaign operators |
| `ascension_foundation_contracts.py` | HCLI/AgentOS foundation contract surfaces |
| `ascension_kernel_registry.py` | kernel/compiler family registry |
| `ascension_knowledge_contract.py` | receipt-bound knowledge-plane schemas |
| `ascension_manager_workflow.py` | dual-manager protocol and lifecycle |
| `ascension_release_workflow.py` | release gates and current product admission |
| `ascension_tournament_workflow.py` | manager tournament protocol |

The compact table preserves the disposition and ownership map. Original
implementations remain recoverable in Git history. The remaining callers were
then audited as one legacy supervisor family: they formed a broken Ascension
V3 controller/gatekeeper/notifier chain around these absent contracts, with no
live production owner and only stale launchd/test wiring. That dependent
family was retired as a unit. The separate `ascension_qwen30_physical_campaign.py`
worker remains because it is an active, independent physical research path.

The same audit found the now-unreferenced `ascension_contracts.py` verifier
also depended on four contract JSONs absent from HEAD. Its 670-line operator
and 191-line test were retired with the dead fixtures; the durable contract
ideas remain represented by the current receipt and manager-protocol surfaces.
