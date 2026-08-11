import torch

from src.training.train_autoencoder import resolve_default_device


def test_resolve_default_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)

    assert resolve_default_device() == "cuda"


def test_resolve_default_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False, raising=False)

    assert resolve_default_device() == "cpu"
