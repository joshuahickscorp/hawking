# HCLI boundary A/B control

This is a transport discriminator, not an Odyssey science task.

Return exactly one valid HCLI structured response with:

- `kind`: `answer`
- `content`: `OK`
- `operations`: `[]`
- `tests`: `[]`
- `tool_calls`: `[]`

Do not call tools, inspect files, mutate files, or add explanation. The
constant being inspected is `7`; the bounded answer is `OK`.
