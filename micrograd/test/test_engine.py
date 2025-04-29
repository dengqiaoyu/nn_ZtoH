import torch
from typing import Final
from micrograd.engine import Value


def test_sanity_check() -> None:
    x: Final[Value] = Value(-4.0)
    z: Final[Value] = 2 * x + 2 + x
    q: Final[Value] = z.relu() + z * x
    h: Final[Value] = (z * z).relu()
    y: Final[Value] = h + q + q * x
    y.backward()
    xmg: Final[Value] = x
    ymg: Final[Value] = y

    x_t: Final[torch.Tensor] = torch.Tensor([-4.0]).double()
    x_t.requires_grad = True
    z_t: Final[torch.Tensor] = 2 * x_t + 2 + x_t
    q_t: Final[torch.Tensor] = z_t.relu() + z_t * x_t
    h_t: Final[torch.Tensor] = (z_t * z_t).relu()
    y_t: Final[torch.Tensor] = h_t + q_t + q_t * x_t
    y_t.backward()
    xpt: Final[torch.Tensor] = x_t
    ypt: Final[torch.Tensor] = y_t

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xpt.grad
    assert xmg.grad == xpt.grad.item()


def test_more_ops() -> None:
    a: Value = Value(-4.0)
    b: Value = Value(2.0)
    c: Value = a + b
    d: Value = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e: Value = c - d
    f: Value = e**2
    g: Value = f / 2.0
    g += 10.0 / f
    g.backward()
    amg: Final[Value] = a
    bmg: Final[Value] = b
    gmg: Final[Value] = g

    a_t: Final[torch.Tensor] = torch.Tensor([-4.0]).double()
    b_t: Final[torch.Tensor] = torch.Tensor([2.0]).double()
    a_t.requires_grad = True
    b_t.requires_grad = True
    c_t: torch.Tensor = a_t + b_t
    d_t: torch.Tensor = a_t * b_t + b_t**3
    c_t = c_t + c_t + 1
    c_t = c_t + 1 + c_t + (-a_t)
    d_t = d_t + d_t * 2 + (b_t + a_t).relu()
    d_t = d_t + 3 * d_t + (b_t - a_t).relu()
    e_t: Final[torch.Tensor] = c_t - d_t
    f_t: Final[torch.Tensor] = e_t**2
    g_t: torch.Tensor = f_t / 2.0
    g_t = g_t + 10.0 / f_t
    g_t.backward()
    apt: Final[torch.Tensor] = a_t
    bpt: Final[torch.Tensor] = b_t
    gpt: Final[torch.Tensor] = g_t

    tol: Final[float] = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert apt.grad
    assert abs(amg.grad - apt.grad.item()) < tol
    assert bpt.grad
    assert abs(bmg.grad - bpt.grad.item()) < tol
