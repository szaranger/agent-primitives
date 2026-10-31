# AI Agent Primitives & Latent Communication MAS

A modular framework and distributed gRPC architecture for Multi-Agent Systems (MAS) implementing reusable Agent Primitives and direct KV-Cache Latent State Communication.

Instead of passing natural-language strings between agents, this project is designed around **latent messages** — raw KV-cache or hidden-state tensors exchanged directly between agents. This removes the lossy decode/re-encode round-trip and reduces token overhead in multi-agent pipelines.

## Why latent communication?

Conventional multi-agent systems communicate via text: Agent A decodes its hidden state into tokens, and Agent B re-encodes those tokens. That round-trip is slow and discards information. By passing tensor state directly, agents can share richer internal representations with far lower latency.

## Core concepts

| Concept | Description |
| --- | --- |
| **Agent Primitive** | A small, reusable building block (reasoner, retriever, planner, critic) that implements a common interface. |
| **`BaseAgent`** | Abstract base class every primitive derives from. Exposes a single `process()` method. |
| **`LatentMessage`** | The unit of inter-agent communication: a tensor state plus optional metadata. |
| **Circuit Breaker** | Async resilience primitive that guards calls to remote agents, preventing cascading failures. |

## Project structure

```
agent-primitives/
├── primitives/
│   └── base.py             # LatentMessage + BaseAgent interface
├── network/
│   └── circuit_breaker.py  # AsyncCircuitBreaker resilience primitive
├── pyproject.toml          # Package metadata & dependencies
└── README.md
```

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/your-org/agent-primitives.git
cd agent-primitives
pip install -e .
```

For development, install the runtime dependencies directly:

```bash
pip install torch transformers grpcio PyJWT redis python-consul
```

## Usage

### Defining an agent primitive

Implement `BaseAgent` and return a `LatentMessage`:

```python
import torch
from primitives.base import BaseAgent, LatentMessage


class ReasonerAgent(BaseAgent):
    def process(self, input_data) -> LatentMessage:
        hidden_state = torch.randn(1, 32, 4096)  # e.g. last-layer KV cache
        return LatentMessage(
            tensor_state=hidden_state,
            metadata={"source": self.name, "kind": "reasoning"},
        )


agent = ReasonerAgent(name="reasoner-1")
msg = agent.process("What is the capital of France?")
print(msg.tensor_state.shape)  # torch.Size([1, 32, 4096])
```

### Consuming a latent message

Downstream agents read `tensor_state` directly and use `metadata` for routing and provenance:

```python
def handle(msg: LatentMessage):
    source = msg.metadata.get("source")
    state = msg.tensor_state
    return state, source
```

### Circuit breaker

`AsyncCircuitBreaker` guards remote agent calls and transitions between `CLOSED`, `OPEN`, and `HALF_OPEN` states when failures exceed a threshold:

```python
from network.circuit_breaker import AsyncCircuitBreaker

breaker = AsyncCircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
```

Configuration:

- `failure_threshold` — number of consecutive failures before the circuit opens.
- `recovery_timeout` — seconds to wait in `OPEN` before allowing a trial request (`HALF_OPEN`).

## Dependencies

| Package | Purpose |
| --- | --- |
| `torch` | Tensor representation of latent state |
| `transformers` | Model loading and hidden-state extraction |
| `grpcio` | Distributed gRPC transport between agents |
| `PyJWT` | Signed authentication of inter-agent messages |
| `redis` | Distributed state and caching |
| `python-consul` | Service discovery for agent endpoints |

## Status & roadmap

This is an early-stage scaffold. The interfaces are stable enough to build against, while the transport layer is under active development.

- [x] `LatentMessage` and `BaseAgent` abstractions
- [x] Async circuit breaker state machine scaffolding
- [ ] Circuit breaker call wrapper (`call()`) and async execution
- [ ] gRPC service definitions (`protos/`) and generated stubs
- [ ] Redis-backed latent state caching
- [ ] Consul service discovery integration
- [ ] JWT-signed inter-agent authentication
- [ ] Reference agent primitives (planner, retriever, critic)

## Contributing

Contributions are welcome. Please open an issue to discuss larger changes before submitting a pull request, and keep new primitives conformant to the `BaseAgent` interface.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
