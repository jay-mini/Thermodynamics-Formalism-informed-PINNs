import torch
import numpy as np
import time
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from src.utils import set_seed, FCNN
from src.choose_optimizer import choose_optimizer


class NM_PINN(torch.nn.Module):
    def __init__(self, X_u_train, q_train, X_f_train, layers, lr, optimizer_name, iterations):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.t_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.q = torch.tensor(q_train, requires_grad=True).float().to(self.device)

        self.k = torch.randn(1, device=self.device, dtype=torch.float32, requires_grad=True)
        self.m = torch.randn(1, device=self.device, dtype=torch.float32, requires_grad=True)
        self.k = torch.nn.Parameter(self.k)
        self.m = torch.nn.Parameter(self.m)

        self.net = FCNN(layers).to(self.device)
        self.net.register_parameter('k', self.k)
        self.net.register_parameter('m', self.m)

        self.optimizer = choose_optimizer(optimizer_name, self.net.parameters(), lr)
        self.optimizer1 = choose_optimizer('LBFGS', self.net.parameters())
        self.iterations = iterations
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.5)
        self.iter = 0
        self.loss, self.loss_q, self.loss_f, self.k_tra, self.m_tra = list(), list(), list(), list(), list()

    def net_q(self, t):
        q = self.net(t)
        return q

    def net_f(self, t):
        q = self.net_q(t)

        q_t = torch.autograd.grad(
            q, t,
            grad_outputs=torch.ones_like(q),
            create_graph=True,
            retain_graph=True,
        )[0]

        q_tt = torch.autograd.grad(
            q_t, t,
            grad_outputs=torch.ones_like(q_t),
            create_graph=True,
            retain_graph=True,
        )[0]

        f = q_tt + self.k / self.m * q

        return f

    def loss_pinn(self, verbose=True):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
            self.optimizer1.zero_grad()
        q_pred = self.net_q(self.t_u)
        f = self.net_f(self.t_f)

        loss_q = torch.mean((self.q - q_pred) ** 2)
        loss_f = torch.mean(f ** 2)

        loss = loss_q + loss_f

        if loss.requires_grad:
            loss.backward()

        grad_norm = 0
        for p in self.net.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_q: %.5e, loss_f: %.5e' % (
                        self.iter, grad_norm, loss.item(), loss_q.item(), loss_f.item())
                )
            self.iter += 1
            self.loss.append(loss.cpu().detach().item())
            self.loss_q.append(loss_q.cpu().detach().item())
            self.loss_f.append(loss_f.cpu().detach().item())
            self.k_tra.append(self.net.k.detach().cpu().numpy())
            self.m_tra.append(self.net.m.detach().cpu().numpy())

        return loss

    def train(self):
        self.net.train()
        for i in range(self.iterations):
            self.optimizer.step(self.loss_pinn)
        print("The training process using adam is finished!")
        self.optimizer1.step(self.loss_pinn)

        return self.loss, self.k_tra, self.m_tra

    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)

        self.net.eval()
        q = self.net_q(t)
        p = torch.autograd.grad(
            q, t,
            grad_outputs=torch.ones_like(q),
            create_graph=True,
            retain_graph=True,
        )[0]
        q = q.detach().cpu().numpy()
        p = p.detach().cpu().numpy()

        return q, p


if __name__ == '__main__':
    set_seed(0)
    iterations = 10000
    N_f = 8192
    t_span = np.array([0, 10])
    k, m, km = 1.0, 1.0, 1.0
    optimizer_name = 'Adam'
    lr = 0.001
    path_test = r'../Data/solution.npz'
    data = np.load(path_test)
    p, q, t = data['arr1'], data['arr2'], data['arr3']
    H = 0.5 * p ** 2 + 0.5 * q ** 2
    dot_q = 1 / m * p
    L = 1 / 2 * m * dot_q ** 2 - 1 / 2 * k * q ** 2

    t_f_train = np.random.uniform(t_span[0], t_span[1], N_f).reshape(-1, 1)
    t_u_train = np.array(t[::50]).reshape(-1, 1)
    q_train = np.array(q[::50]).reshape(-1, 1)
    print(q_train.shape)

    layers = [1, 64, 64, 64, 64, 1]
    model = NM_PINN(t_u_train, q_train, t_f_train, layers, lr, optimizer_name, iterations)
    start_time = time.time()
    # loss = model.train()
    end_time = time.time()

    # torch.save(model.state_dict(), r'../Data/Newton_PINNs.pth')
    model.load_state_dict(torch.load(f'../Data/Newton_PINNs.pth'))

    q_pred, dq_pred = model.predict(t.reshape(-1, 1))
    error_q_relative = np.linalg.norm(q.reshape(-1, 1) - q_pred.reshape(-1, 1), 2) / np.linalg.norm(q.reshape(-1, 1), 2)

    k_value = model.k.detach().cpu().numpy()
    m_value = model.m.detach().cpu().numpy()
    km_value = k_value / m_value
    error_km = np.abs(km_value - km) / km
    p_pred = dq_pred * m_value
    H_pred = 0.5 * q_pred ** 2 + 0.5 * p_pred.reshape(-1, 1) ** 2
    L_pred = 1 / 2 * m_value * dq_pred ** 2 - 1 / 2 * k_value * q_pred ** 2
    error_H_relative = np.linalg.norm(H.reshape(-1, 1) - H_pred.reshape(-1, 1), 2) / np.linalg.norm(H.reshape(-1, 1), 2)
    error_L_relative = np.linalg.norm(L.reshape(-1, 1) - L_pred.reshape(-1, 1), 2) / np.linalg.norm(L.reshape(-1, 1), 2)
    print('Error q: %.5e, Error_km: %.5e' % (error_q_relative, error_km))
    print('Error H: %.5e, Error L: %.5e' % (error_H_relative, error_L_relative))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['svg.fonttype'] = 'none'

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    axs[0].set_xlim(0.0, 10)
    axs[0].plot(t, H, label='True', linewidth=5)
    axs[0].plot(t, H_pred, label='NM PINNs', color='darkorange', marker='*',
                markersize=12, markevery=200, linewidth=5, linestyle='--')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel('$Time$', fontsize=18, fontweight='bold')
    axs[0].set_ylabel('$H$', fontsize=18, fontweight='bold')
    axs[0].tick_params(labelsize=18)
    axs[0].legend(loc='upper left', fontsize=22)

    axs[1].set_xlim(0.0, 10)
    axs[1].plot(t, L, linewidth=5)
    axs[1].plot(t, L_pred, color='darkorange', marker='*',
                markersize=12, markevery=40, linewidth=5, linestyle='--')
    axs[1].set_xlabel('$Time$', fontsize=18, fontweight='bold')
    axs[1].set_ylabel('$L$', fontsize=18, fontweight='bold')
    axs[1].tick_params(labelsize=18)

    axs[2].set_xlim(0.0, 10)
    axs[2].plot(t, q, linewidth=5)
    axs[2].plot(t, q_pred, marker='*', markersize=12, linestyle='--', color='darkorange', markevery=40, linewidth=5)
    axs[2].set_xlabel('$Time$', fontsize=18, fontweight='bold')
    axs[2].set_ylabel('$q$', fontsize=18, fontweight='bold')
    axs[2].tick_params(labelsize=18)

    axs[3].plot(q, p, linewidth=5)
    axs[3].plot(q_pred, p_pred, marker='*', markersize=12,
                color='darkorange', markevery=40, linewidth=5, linestyle='--')
    axs[3].set_xlabel('$q$', fontsize=18, fontweight='bold')
    axs[3].set_ylabel('$p$', fontsize=18, fontweight='bold')
    axs[3].tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()
    fig.savefig('figure_1.svg', dpi=600)
