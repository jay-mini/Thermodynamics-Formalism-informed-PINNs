import torch
import numpy as np
import time
from matplotlib import pyplot as plt
from src.utils import set_seed, FCNN
from src.choose_optimizer import choose_optimizer


class OVP_PINN(torch.nn.Module):
    def __init__(self, X_u_train, u_train, X_f_train, layers, lr, optimizer_name, iterations):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.t_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(u_train, requires_grad=True).float().to(self.device)

        self.lamb = torch.randn(1, device=self.device, dtype=torch.float32, requires_grad=True)
        self.beta = torch.randn(1, device=self.device, dtype=torch.float32, requires_grad=True)
        self.lamb = torch.nn.Parameter(self.lamb)
        self.beta = torch.nn.Parameter(self.beta)

        self.net = FCNN(layers)
        self.net.register_parameter('lamb', self.lamb)
        self.net.register_parameter('beta', self.beta)
        self.net.to(self.device)

        self.optimizer = choose_optimizer(optimizer_name, self.net.parameters(), lr)
        self.optimizer1 = choose_optimizer('LBFGS', self.net.parameters())
        self.iterations = iterations
        self.iter = 0
        self.loss, self.loss_u, self.loss_f, self.lamb_tra, self.beta_tra = list(), list(), list(), list(), list()

    def net_u(self, t):
        u = self.net(t)
        return u

    def net_R(self, u, w, w_t):
        return 0.5 * self.lamb * w ** 2 + self.beta ** 2 * u * w + w * w_t

    def net_r(self, t):
        result = self.net_u(t)

        q, w = result[:, 0:1], result[:, 1:2]

        q_t = torch.autograd.grad(
            q, t,
            grad_outputs=torch.ones_like(q),
            retain_graph=True,
            create_graph=True,
        )[0]

        w_t = torch.autograd.grad(
            w, t,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True,
        )[0]

        R = self.net_R(q, w, w_t)

        R_w = torch.autograd.grad(
            R, w,
            grad_outputs=torch.ones_like(R),
            retain_graph=True,
            create_graph=True,
        )[0]

        f = q_t - w

        return R_w, f

    def loss_pinn(self, verbose=True):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
            self.optimizer1.zero_grad()
        u_pred = self.net_u(self.t_u)
        dr, f = self.net_r(self.t_f)

        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_dr = torch.mean(dr ** 2)
        loss_f = torch.mean(f ** 2)

        loss = loss_u + loss_dr + 10*loss_f

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
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_u: %.5e, loss_f: %.5e, loss_dr: %.5e' % (
                        self.iter, grad_norm, loss.item(), loss_u.item(), loss_f.item(), loss_dr.item())
                )
            self.iter += 1
            self.loss.append(loss.cpu().detach().item())
            self.loss_u.append(loss_u.cpu().detach().item())
            self.loss_f.append(loss_f.cpu().detach().item())
            self.lamb_tra.append(self.net.lamb.detach().cpu().numpy())
            self.beta_tra.append(self.net.beta.detach().cpu().numpy())

        return loss

    def train(self):
        self.net.train()
        for i in range(self.iterations):
            self.optimizer.step(self.loss_pinn)
            self.scheduler.step()
        print("The training process using adam is finished!")
        self.optimizer1.step(self.loss_pinn)

        return self.loss, self.lamb_tra, self.beta_tra

    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)

        self.net.eval()
        result = self.net_u(t)

        q, w = result[:, 0:1], result[:, 1:2]

        w_t = torch.autograd.grad(
            w, t,
            grad_outputs=torch.ones_like(w),
            retain_graph=True,
            create_graph=True,
        )[0]

        q = q.detach().cpu().numpy()
        w = w.detach().cpu().numpy()
        w_t = w_t.detach().cpu().numpy()

        return q, w, w_t


if __name__ == '__main__':
    set_seed(0)
    iterations = 50000
    N_f = 8192
    t_span = np.array([0, 10])
    gamma, m, l, g = 0.2, 1.0, 1.0, 9.81
    lamb, beta = gamma / m, (g / l) ** 0.5
    optimizer_name = 'Adam'
    lr = 0.005
    path_test = r'../Data/solution.npz'
    data = np.load(path_test)
    t, q, w, R = data['arr1'], data['arr2'], data['arr3'], data['arr4']
    S = -0.5 * beta ** 2 * q ** 2 - 0.5 * w ** 2

    t_f_train = np.random.uniform(t_span[0], t_span[1], N_f).reshape(-1, 1)
    t_u_train = np.array(t[::10]).reshape(-1, 1)
    u_train = np.concatenate((q[::10].reshape(-1, 1), w[::10].reshape(-1, 1)), axis=1)
    print(u_train.shape)

    layers = [1, 64, 64, 64, 64, 2]
    model = OVP_PINN(t_u_train, u_train, t_f_train, layers, lr, optimizer_name, iterations)
    start_time = time.time()
    # loss = model.train()
    end_time = time.time()

    # torch.save(model.state_dict(), r'../Data/Onsager_PINNs.pth')
    model.load_state_dict(torch.load(f'../Data/Onsager_PINNs.pth'))

    u_pred, w_pred, w_t_pred = model.predict(t.reshape(-1, 1))
    error_u_relative = np.linalg.norm(q.reshape(-1, 1) - u_pred.reshape(-1, 1), 2) / np.linalg.norm(
        q.reshape(-1, 1))

    pred_lamb = model.lamb.detach().cpu().numpy()
    pred_beta = model.beta.detach().cpu().numpy()
    R_pred = 0.5 * pred_lamb * w_pred ** 2 + pred_beta ** 2 * u_pred * w_pred + w_pred * w_t_pred
    S_pred = -0.5 * pred_beta ** 2 * u_pred ** 2 - 0.5 * w_pred ** 2

    error_lamb = np.linalg.norm(pred_lamb - lamb, 2) / np.linalg.norm(lamb)
    error_beta = np.linalg.norm(pred_beta - beta, 2) / np.linalg.norm(beta)
    error_R_relative = np.linalg.norm(R.reshape(-1, 1) - R_pred.reshape(-1, 1), 2) / np.linalg.norm(R.reshape(-1, 1), 2)
    error_S_relative = np.linalg.norm(S.reshape(-1, 1) - S_pred.reshape(-1, 1), 2) / np.linalg.norm(S.reshape(-1, 1), 2)

    print('Error u: %.5e, Error lamb: %.5e, Error beta: %.5e' % (
        error_u_relative, error_lamb, error_beta))
    print('Error R: %.5e, Error S: %.5e' % (error_R_relative, error_S_relative))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['svg.fonttype'] = 'none'

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    axs[0].set_xlim(0.0, 10)
    axs[0].plot(t, R, label='True', linewidth=5)
    axs[0].plot(t, R_pred, label='OVP PINNs', color='darkorange', marker='*',
                markersize=12, markevery=200, linewidth=5, linestyle='--')
    axs[0].set_xlabel('$Time$', fontsize=18, fontweight='bold')
    axs[0].set_ylabel('$R$', fontsize=18, fontweight='bold')
    axs[0].tick_params(labelsize=18)
    axs[0].legend(loc='upper left', fontsize=22)

    axs[1].set_xlim(0.0, 10)
    axs[1].plot(t, S, linewidth=5)
    axs[1].plot(t, S_pred, color='darkorange', marker='*',
                markersize=12, markevery=200, linewidth=5, linestyle='--')
    axs[1].set_xlabel('$Time$', fontsize=18, fontweight='bold')
    axs[1].set_ylabel('$S$', fontsize=18, fontweight='bold')
    axs[1].tick_params(labelsize=18)

    axs[2].set_xlim(0.0, 10)
    axs[2].plot(t, q, linewidth=5)
    axs[2].plot(t, u_pred, marker='*', markersize=12, linestyle='--', color='darkorange', markevery=40, linewidth=5)
    axs[2].set_xlabel('$Time$', fontsize=18, fontweight='bold')
    axs[2].set_ylabel('$q$', fontsize=18, fontweight='bold')
    axs[2].tick_params(labelsize=18)

    axs[3].plot(q, w, linewidth=5)
    axs[3].plot(u_pred, w_pred, marker='*', markersize=12,
                color='darkorange', markevery=40, linewidth=5, linestyle='--')
    axs[3].set_xlabel('$q$', fontsize=18, fontweight='bold')
    axs[3].set_ylabel('$p$', fontsize=18, fontweight='bold')
    axs[3].tick_params(labelsize=18)

    plt.tight_layout()
    plt.show()
    fig.savefig('figure_2.svg', dpi=600)
