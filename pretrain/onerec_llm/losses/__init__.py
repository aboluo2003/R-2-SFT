from onerec_llm.losses.registry import get_loss_components

_DEFAULT_COMPONENTS = {}


def __getattr__(name):
  if name in {"CrossEntropyLoss", "ChunkedLossComputer"}:
    if not _DEFAULT_COMPONENTS:
      loss_cls, chunked_cls = get_loss_components("ce")
      _DEFAULT_COMPONENTS.update(
        CrossEntropyLoss=loss_cls,
        ChunkedLossComputer=chunked_cls,
      )
    return _DEFAULT_COMPONENTS[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
  "CrossEntropyLoss",
  "ChunkedLossComputer",
  "get_loss_components",
]
