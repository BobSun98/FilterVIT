import torch


def generate_random_bool_tensor(batch, n, a):
    # 创建 batch * n * n 的全 False tensor
    bool_tensor = torch.zeros((batch, n * n), dtype=torch.bool)

    # 为每个 batch 生成 a 个随机的 True 位置
    indices = torch.rand(batch, n * n).argsort(dim=1)[:, :a]

    # 将这些位置设置为 True
    bool_tensor.scatter_(1, indices, True)

    # reshape 为 [batch, n, n]
    bool_tensor = bool_tensor.view(batch, n, n)

    return bool_tensor


# 使用例子
batch = 5
n = 4
a = 5
random_bool_tensor = generate_random_bool_tensor(batch, n, a)
print(random_bool_tensor)