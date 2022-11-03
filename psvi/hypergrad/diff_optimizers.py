# Copyright (c) Meta Platforms, Inc. and affiliates.
# All Rights Reserved.


from itertools import repeat

import torch


class DifferentiableOptimizer:
    def __init__(self, loss_f, dim_mult, data_or_iter=None):
        """
        Args:
            loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
            data_or_iter: (x, y) or iterator over the data needed for loss_f
        """
        self.data_iterator = None
        if data_or_iter:
            self.data_iterator = (
                data_or_iter
                if hasattr(data_or_iter, "__next__")
                else repeat(data_or_iter)
            )

        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def get_opt_params(self, params):
        opt_params = list(params)
        for _ in range(self.dim_mult - 1):
            opt_params.extend([torch.zeros_like(p) for p in params])
        return opt_params

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

    def get_loss(self, params, hparams):
        if self.data_iterator:
            data = next(self.data_iterator)
            self.curr_loss = self.loss_f(params, hparams, data)
        else:
            self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss


class GradientDescent(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, data_or_iter=None):
        super(GradientDescent, self).__init__(
            loss_f, dim_mult=1, data_or_iter=data_or_iter
        )
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)
        return gd_step(params, loss, sz, create_graph=create_graph)


class HeavyBall(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, momentum, data_or_iter=None):
        super(HeavyBall, self).__init__(loss_f, dim_mult=2, data_or_iter=data_or_iter)
        self.loss_f = loss_f
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size
        self.momentum_f = momentum if callable(momentum) else lambda x: momentum

    def step(self, params, hparams, create_graph):
        n = len(params) // 2
        p, p_aux = params[:n], params[n:]
        loss = self.get_loss(p, hparams)
        sz, mu = self.step_size_f(hparams), self.momentum_f(hparams)
        p_new, p_new_aux = heavy_ball_step(
            p, p_aux, loss, sz, mu, create_graph=create_graph
        )
        return [*p_new, *p_new_aux]


class Momentum(DifferentiableOptimizer):
    """
    GD with momentum step as implemented in torch.optim.SGD
    .. math::
              v_{t+1} = \mu * v_{t} + g_{t+1} \\
              p_{t+1} = p_{t} - lr * v_{t+1}
    """

    def __init__(self, loss_f, step_size, momentum=0.9, data_or_iter=None):
        super(Momentum, self).__init__(loss_f, dim_mult=2, data_or_iter=data_or_iter)
        self.loss_f = loss_f
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size
        self.momentum_f = momentum if callable(momentum) else lambda x: momentum

    def step(self, params, hparams, create_graph):
        n = len(params) // 2
        p, p_aux = params[:n], params[n:]
        loss = self.get_loss(p, hparams)
        sz, mu = self.step_size_f(hparams), self.momentum_f(hparams)
        p_new, p_new_aux = torch_momentum_step(
            p, p_aux, loss, sz, mu, create_graph=create_graph
        )
        return [*p_new, *p_new_aux]


class DifferentiableAdam(DifferentiableOptimizer):
    """
    DifferentiableAdam optimizer as implemented in torch.optim.Adam
    .. math::

              m_{t+1} = beta_1 * m_{t} + (1 - beta1) * g_{t}
              u_{t+1} = beta_2 * u_{t} + (1 - beta2) * g_{t}^2
              mh_{t+1} = mh_{t+1} / (1 - beta1**t)
              uh_{t+1} = uh_{t+1} / (1 - beta2**t)
              p_{t+1} = p_{t} - lr * mh_{t+1} / (sqrt(uh_{t+1} + eps))
    """

    def __init__(
        self,
        loss_f,
        step_size,
        data_or_iter=None,
        betas=(0.9, 0.999),
        eps=1e-8,
        step_cnt=1,
    ):
        super(DifferentiableAdam, self).__init__(
            loss_f, dim_mult=3, data_or_iter=data_or_iter
        )
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size
        (self.beta1, self.beta2) = betas
        self.eps = eps
        self.step_cnt = step_cnt

    def step(self, params, hparams, create_graph):
        n = len(params) // 3
        p, m, u = params[:n], params[n : 2 * n], params[2 * n :]
        loss = self.get_loss(p, hparams)
        sz = self.step_size_f(hparams)
        p_new, m_new, u_new = adam_step(
            p,
            m,
            u,
            loss,
            sz,
            self.step_cnt,
            self.beta1,
            self.beta2,
            self.eps,
            create_graph=create_graph,
        )
        self.step_cnt += 1
        return [*p_new, *m_new, *u_new]


def gd_step(params, loss, step_size, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g for w, g in zip(params, grads)]


def heavy_ball_step(params, aux_params, loss, step_size, momentum, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [
        w - step_size * g + momentum * (w - v)
        for g, w, v in zip(grads, params, aux_params)
    ], params


def torch_momentum_step(
    params, aux_params, loss, step_size, momentum=0.9, create_graph=True
):
    """
    GD with momentum step as implemented in torch.optim.SGD
    .. math::
              v_{t+1} = \mu * v_{t} + g_{t+1} \\
              p_{t+1} = p_{t} - lr * v_{t+1}
    """
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    new_aux_params = [momentum * v + g for v, g in zip(aux_params, grads)]
    return [w - step_size * nv for w, nv in zip(params, new_aux_params)], new_aux_params


def adam_step(
    params,
    ms,
    us,
    loss,
    step_size,
    step_cnt,
    beta1,
    beta2,
    eps,
    momentum=0.9,
    create_graph=True,  # False when used with approximate implicit gradient; should be True otherwise
):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    new_m = [beta1 * m + (1.0 - beta1) * g for m, g in zip(ms, grads)]
    new_u = [beta2 * u + (1.0 - beta2) * g**2 + 1e-12 for u, g in zip(us, grads)]
    return (
        [
            w
            - step_size
            * (
                m
                / (1.0 - beta1**step_cnt)
                / (torch.sqrt(u / (1 - beta2**step_cnt)) + eps)
            )
            for w, m, u in zip(params, new_m, new_u)
        ],
        new_m,
        new_u,
    )
