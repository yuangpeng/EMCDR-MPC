import pandas as pd
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    """ import data
    input:
        file_path -> file path of the rating date
    output:
        ui_rating.values -> rating matrix
    """
    ui_rating = pd.read_csv(file_path, index_col=0).fillna(0)
    return ui_rating.values


def matrix_factorization(rating_matrix, k, learning_rate, beta, training_epochs, display_step=100):
    """ matrix factorization using gradient descent
    input:
        data: rating matrix
        k(int): 分解矩阵的参数
        learning_rate(float): 学习率
        beta(float): 正则化参数
        training_epochs(int): 最大迭代次数
    output:
        U, V: 分解后的矩阵
    """
    m, n = np.shape(rating_matrix)

    # initialize feature matrix of U and V
    U = torch.tensor(np.random.normal(0, 0.1, size=(m, k)), requires_grad=True, dtype=torch.float, device=device)
    V = torch.tensor(np.random.normal(0, 0.1, size=(n, k)), requires_grad=True, dtype=torch.float, device=device)

    for epoch in range(training_epochs):
        l2_reg = beta * (torch.pow(U, 2).sum() + torch.pow(V, 2).sum()).to(device)
        criterion = torch.nn.MSELoss(reduction='mean')
        loss = criterion(torch.matmul(U, torch.transpose(V, 0, 1)), torch.tensor(rating_matrix, dtype=torch.float, device=device)) + l2_reg
        loss.backward()

        if (epoch + 1) % display_step == 0:
            print(f'Epoch: {epoch + 1}, cost = {loss}')

        if (epoch + 1) % 50000 == 0:
            torch.save(U, f'./UI/U_{epoch + 1}.pt')
            torch.save(V, f'./UI/V_{epoch + 1}.pt')

        with torch.no_grad():
            U -= learning_rate * U.grad
            V -= learning_rate * V.grad

            U.grad = None
            V.grad = None

    print('Variable: U')
    print(f'Shape: {U.shape}')
    print(U)
    print('Variable: V')
    print(f'Shape: {V.shape}')
    print(V)
    print("Optimization Finished!")


if __name__ == "__main__":
    data = load_data("./data/t_rate.csv")
    matrix_factorization(data, 5, 0.0002, 0.02, 400000, 1000)
