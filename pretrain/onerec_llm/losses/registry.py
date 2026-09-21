import importlib
from typing import Tuple


LOSS_IMPL_TO_MODULE = {
    "ce": "onerec_llm.losses.ce",
    "ce_v2": "onerec_llm.losses.ce_v2",
}


def resolve_loss_module_name(loss_impl: str) -> str:
    try:
        return LOSS_IMPL_TO_MODULE[loss_impl]
    except KeyError as exc:
        supported = ", ".join(sorted(LOSS_IMPL_TO_MODULE))
        raise ValueError(
            f"Unsupported loss_impl={loss_impl!r}. Supported values: {supported}"
        ) from exc


def get_loss_components(loss_impl: str) -> Tuple[type, type]:
    module = importlib.import_module(resolve_loss_module_name(loss_impl))
    return module.CrossEntropyLoss, module.ChunkedLossComputer
