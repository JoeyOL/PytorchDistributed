import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    # nccl：NVIDIA Collective Communication Library 
    # 分布式情况下的，gpus 间通信
    # torchrun 会自动设置 rank 和 world_size，也不用设置 MASTER_ADDR 和 MASTER_PORT，
    # 可以直接在torchrun的命令行中设置
    
    # 1. DDP必须要设置的两件事
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 ) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        # 2. 用于分布式训练的模型必须要用DDP包裹
        # 如果想要获取DDP包裹的model内部方法，需要调用self.module.module
        self.model = DDP(model, device_ids=[self.gpu_id])
    
    def _run_batch(self, xs, ys):
        self.optimizer.zero_grad()
        output = self.model(xs)
        loss = F.cross_entropy(output, ys)
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f'[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)}')
        # 3. 设置epoch，DDP会根据epoch来划分数据集
        self.train_dataloader.sampler.set_epoch(epoch)
        for xs, ys in self.train_dataloader:
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            self._run_batch(xs, ys)
    
    def train(self, max_epoch: int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)
            
class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
            
def main(max_epochs: int, batch_size: int):
    ddp_setup()
    
    train_dataset = MyTrainDataset(2048)
    # 4. 使用DistributedSampler来划分数据集
    sampler = DistributedSampler(train_dataset, shuffle=True)
    # pin_memory=True: 让数据集在GPU上预分配内存，避免每次都要从CPU拷贝到GPU
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  pin_memory=True, 
                                  sampler=sampler)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model=model, optimizer=optimizer, train_dataloader=train_dataloader)
    trainer.train(max_epochs)
    
    destroy_process_group()

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='DDP Example')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # 运行torchrun时一共会启动n个进程，并且会给每个进程分配一个rank
    # rank是从0开始的，表示当前进程的编号
    # parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    
    args = parser.parse_args()
    # torchrun会自动设置rank和world_size，所以这里不需要手动设置
    print(f"local_rank: {int(os.environ["LOCAL_RANK"])}, world_size: {torch.cuda.device_count()}")
    
    main(args.max_epochs, args.batch_size)
    # 运行命令：
    # torchrun --nproc_per_node=2 ddp_gpus_torchrun.py --max_epochs 10 --batch_size 32