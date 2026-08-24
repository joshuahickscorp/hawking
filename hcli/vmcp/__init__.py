"""VMCP: sensory / evidence tools.

VisionMCP is already its own installable package under ``visionmcp/``.
Harnesses that need it insert ``visionmcp/src`` (a genuine foreign
package root, not a second name for hcli). This marker exists so the
control plane does not grow a parallel ``hcli.vmcp.*`` implementation.
"""

OWNED_PREFIXES = ("visionmcp/",)
