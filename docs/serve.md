# HTTP API Reference

`hawking serve` exposes an OpenAI-compatible HTTP API on the address you
specify. All endpoints are synchronous or SSE-streaming; no WebSocket.

## Starting the server

```sh
hawking serve \
  --weights models/deepseek-v2-lite-q4.gguf \
  --addr 127.0.0.1:8080
```

The server is ready when you see a log line containing `listening`. Model
loading takes 10–60 seconds depending on hardware and page cache warmth. Poll
`/healthz` to detect readiness:

```sh
until curl -sf http://127.0.0.1:8080/healthz; do sleep 1; done
echo "server ready"
```

## Endpoints

### GET /healthz

Returns `ok` (plain text, HTTP 200) when the server is alive.

```sh
curl http://127.0.0.1:8080/healthz
# ok
```

### GET /v1/models

Lists the loaded model.

```sh
curl http://127.0.0.1:8080/v1/models
```

```json
{
  "object": "list",
  "data": [{"id": "DeepSeek-V2-Lite-Chat", "object": "model"}]
}
```

The model `id` comes from the GGUF `general.name` metadata field.

### POST /v1/chat/completions

OpenAI chat completions. Supports non-streaming (default) and SSE streaming.

**Request fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | — | Any value accepted; model is fixed at load time |
| `messages` | array | required | `[{"role": "user\|assistant\|system", "content": "..."}]` |
| `max_tokens` | int | 256 | Max new tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0 = greedy) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `seed` | int | null | Optional RNG seed for reproducibility |
| `stream` | bool | false | If true, returns SSE stream |

**Non-streaming example:**

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "Write a haiku about Metal kernels."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

```json
{
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Threads of light ignite,\nSilicon weaves the dream fast,\nGPU takes flight."}
  }]
}
```

**Streaming example:**

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V2-Lite-Chat","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"stream":true}'
```

Each SSE event is `data: <json>` where the JSON has
`choices[0].delta.content`. The stream ends with `data: [DONE]`.

**Python (openai SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="DeepSeek-V2-Lite-Chat",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=16,
    temperature=0,
)
print(response.choices[0].message.content)
```

### POST /v1/completions

Legacy text completions (no chat template).

**Request fields:** same as chat completions but `prompt` (string) replaces
`messages`.

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V2-Lite-Chat","prompt":"Once upon a time","max_tokens":32,"temperature":0}'
```

```json
{
  "object": "text_completion",
  "choices": [{"index": 0, "text": " there was a small village nestled in the mountains..."}]
}
```

### GET /metrics

Prometheus text format metrics stub. Real metrics arrive in v0.2.

```sh
curl http://127.0.0.1:8080/metrics
# # hawking_metrics 1
```

## Chat template

The current chat template wraps each message as `<|role|>\ncontent\n` and
appends `<|assistant|>\n`. Full Jinja2 chat-template support (reading the
template from GGUF metadata) is a v0.2 item.

## Known limitations

- Single concurrent request. A second request queues behind the first; there
  is no batching or slot manager yet (v0.2 item).
- No auth / API key enforcement. Run behind a reverse proxy if exposed beyond
  localhost.
- No usage token counts in the response body yet.
