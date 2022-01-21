import torch
import torch.nn as nn
import os
import random
import torch.optim as optim
import time
from torchvision import datasets, transforms
from preactresnet import PreActResNet18
from dataset import cifar100
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

iterations = 20

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

def get_device_id():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args.local_rank

def save_imgs(t, dst_dir, img_path):
    toPIL = T.ToPILImage()
    bs = t.shape[0]
    for i in range(bs):
        img = toPIL(t[i].detach().cpu())
        dir_name = img_path[i].split("/")[-2]
        file_name = img_path[i].split("/")[-1]
        img_dir = os.path.join(dst_dir, dir_name)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img.save(os.path.join(img_dir, file_name))

def rebuild_image_fv_bn(self, image, model, randstart=True):
        model.eval()
        model.set_hook()

        with torch.no_grad():
            ori_fv = model.fv(image)

        def criterion(x, y):
            rnd_fv = model.fv(x)
            return torch.div(torch.norm(rnd_fv - ori_fv, dim=1), torch.norm(ori_fv, dim=1)).mean()

        if randstart == True:
            if len(image.shape) == 3:
                rand_x = torch.rand_like(image.unsqueeze(0), requires_grad=True, device=self._device)
            else:
                rand_x = torch.rand_like(image, requires_grad=True, device=self._device)

        start_time = time.time()
        lr = 0.01
        r_feature = 1e-3

        lim_0 = 10
        lim_1 = 10
        var_scale_l2 = 1e-4
        var_scale_l1 = 0.0
        l2_scale = 1e-5
        first_bn_multiplier = 1

        best_img = None
        optimizer = optim.Adam([rand_x], lr=lr, betas=[0.5, 0.9], eps=1e-8)
        for i in range(iterations):
            # roll
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(rand_x, shifts=(off1, off2), dims=(2, 3))

            # R_prior losses
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
            # l2 loss on images
            loss_l2 = torch.norm(inputs_jit.view(inputs_jit.shape[0], -1), dim=1).mean()
            # main loss
            main_loss = criterion(inputs_jit, torch.tensor([0]))

            # bn loss
            if iterations == 600:
                if i <= 200:
                    r_feature = 1e-3
                elif i <= 400:
                    r_feature = 1e-2
                elif i <= 600:
                    r_feature = 5e-2
            elif iterations == 2000:
                if i <= 500:
                    r_feature = 1e-3
                elif i <= 1200:
                    r_feature = 5e-3
                elif i <= 2000:
                    r_feature = 1e-2

            rescale = [first_bn_multiplier] + [1. for _ in range(len(model.loss_r_feature_layers) - 1)]
            loss_r_feature = sum(
                [rescale[idx] * item.r_feature for idx, item in enumerate(model.loss_r_feature_layers)])
            loss = main_loss + r_feature * loss_r_feature + var_scale_l2 * loss_var_l2 + var_scale_l1 * loss_var_l1 + l2_scale * loss_l2

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            rand_x.data = torch.clamp(rand_x.data, 0, 1)
            # if (i+1) % 100 == 0 :
            #     print(i, 'rand_strat ', randstart)
            #     print(
            #         f'loss {loss:.3f} , fv_loss {main_loss:.3f} , loss_r_fea {loss_r_feature:.3f} , loss_l2 {loss_l2:.3f} , loss_var_l2 {loss_var_l2:.3f}')

            best_img = rand_x.clone().detach()
        print("inverse --- %s seconds ---" % (time.time() - start_time))
        model.remove_hook()
        return best_img

if __name__ == "__main__":
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    bs = 64

    train_trsf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255),
        transforms.ToTensor(),
    ])

    test_trsf = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataroot = "/data/junjie/results/cifar100_png"

    train_dataset = cifar100(dataroot, transform=train_trsf, train=True, inversion=True)
    # test_dataset = cifar100(dataroot, transform=test_trsf, train=False, inversion=False)

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=12, sampler=train_sampler, shuffle=False, pin_memory=True)

    net = PreActResNet18(num_classes=100)
    pretrained_dict=torch.load("/home/20/junjie/code/zhengjin/pretrained_robust_model/cifar100_linf_eps8.pth")
    model_dict=net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "linear" not in k}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    net = nn.parallel.DistributedDataParallel(net, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)

    dst_dir = "/data/junjie/results/cifar100_inverse/train"
    for train_data, _, img_path in tqdm(train_loader):
        r_data = rebuild_image_fv_bn(train_data, net.module, True)
        save_imgs(r_data, dst_dir, img_path)