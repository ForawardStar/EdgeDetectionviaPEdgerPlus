import argparse

import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import *
from loss_function import MyLoss
from models_recurrent import *
from models_nonrecurrent import Net_NonRecurrent
from models_recurrent_bayesian import *
from models_nonrecurrent_bayesian import Net_NonRecurrent_bayesian
from tools import SingleSummaryWriter, mutils, saver
from tools.metric_utils import AverageMeters, write_loss
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--comment', '-m', default='edge_detection')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--iter_size', type=int, default=16, help='size of the iterations')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=20, help='number of cpu threads to use during batch generation')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model checkpoints')
parser.add_argument("--log_interval", type=int, default=500, help="interval for logging")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--log_path', type=str, default='logs/')
parser.add_argument('--saved_path', type=str, default='logs/')


# ----------
#  Training
# ----------
def main():
    global global_step
    global W1_recurrent_previous
    global W2_recurrent_previous
    global W3_recurrent_previous

    global W1_nonrecurrent_previous
    global W2_nonrecurrent_previous
    global W3_nonrecurrent_previous


    for epoch in range(args.epoch, args.n_epochs):
        if epoch >= 2:
            state_st = G_network_recurrent.state_dict()
            state_t = G_network_recurrent_teacher.state_dict()
            for k, v in state_t.items():
                state_t[k] = (state_t[k] + state_st[k]) * 0.5
            G_network_recurrent_teacher.load_state_dict(state_t)

            state_st_nonrecurrent = G_network_nonrecurrent.state_dict()
            state_t_nonrecurrent = G_network_nonrecurrent_teacher.state_dict()
            for k, v in state_t_nonrecurrent.items():
                state_t_nonrecurrent[k] = (state_t_nonrecurrent[k] + state_st_nonrecurrent[k]) * 0.5
            G_network_nonrecurrent_teacher.load_state_dict(state_t_nonrecurrent)

        elif epoch == 1:
            G_network_recurrent_teacher.load_state_dict(G_network_recurrent.state_dict())

            G_network_nonrecurrent_teacher.load_state_dict(G_network_nonrecurrent.state_dict())

        if epoch >= 1:
            # Update the parameters of Bayesian Networks
            state_t = G_network_recurrent_teacher.state_dict()

            state_b1 = G_network_recurrent_bayesian1.state_dict()
            for k, v in state_t.items():
                state_b1[k] = state_t[k]
            G_network_recurrent_bayesian1.load_state_dict(state_b1)

            state_b2 = G_network_recurrent_bayesian2.state_dict()
            for k, v in state_t.items():
                state_b2[k] = state_t[k]
            G_network_recurrent_bayesian2.load_state_dict(state_b2)

            state_b3 = G_network_recurrent_bayesian3.state_dict()
            for k, v in state_t.items():
                state_b3[k] = state_t[k]
            G_network_recurrent_bayesian3.load_state_dict(state_b3)

            ##################################################################333
            state_t_nonrecurrent = G_network_nonrecurrent_teacher.state_dict()

            state_b1_nonrecurrent = G_network_nonrecurrent_bayesian1.state_dict()
            for k, v in state_t_nonrecurrent.items():
                state_b1_nonrecurrent[k] = state_t_nonrecurrent[k]
            G_network_nonrecurrent_bayesian1.load_state_dict(state_b1_nonrecurrent)

            state_b2_nonrecurrent = G_network_nonrecurrent_bayesian2.state_dict()
            for k, v in state_t_nonrecurrent.items():
                state_b2_nonrecurrent[k] = state_t_nonrecurrent[k]
            G_network_nonrecurrent_bayesian2.load_state_dict(state_b2_nonrecurrent)

            state_b3_nonrecurrent = G_network_nonrecurrent_bayesian3.state_dict()
            for k, v in state_t_nonrecurrent.items():
                state_b3_nonrecurrent[k] = state_t_nonrecurrent[k]
            G_network_nonrecurrent_bayesian3.load_state_dict(state_b3_nonrecurrent)
      
           
            print("Finish sampling parameters")
            # Computing weights of every parameter samplings 
            bar_val = tqdm.tqdm(val_dataloader, disable=True)
            val_length = len(bar_val)
            constants = 0.0
            predictions_recurrent1 = 0.0
            predictions_recurrent2 = 0.0
            predictions_recurrent3 = 0.0
            predictions_nonrecurrent1 = 0.0
            predictions_nonrecurrent2 = 0.0
            predictions_nonrecurrent3 = 0.0
            for i, val_batch in enumerate(bar_val):
                val_img = val_batch['img'].float().to(device)
                val_edge_gt = val_batch['edge'].float().to(device)

                M_recurrent = W1_recurrent_previous * torch.sigmoid(G_network_recurrent_bayesian1(val_img)[-1]) + W2_recurrent_previous * torch.sigmoid(G_network_recurrent_bayesian2(val_img)[-1]) + W3_recurrent_previous * torch.sigmoid(G_network_recurrent_bayesian3(val_img)[-1])
                M_nonrecurrent = W1_nonrecurrent_previous * torch.sigmoid(G_network_nonrecurrent_bayesian1(val_img)[-1]) + W2_nonrecurrent_previous * torch.sigmoid(G_network_nonrecurrent_bayesian2(val_img)[-1]) + W3_nonrecurrent_previous * torch.sigmoid(G_network_nonrecurrent_bayesian3(val_img)[-1])

                constants = constants + torch.mean(val_edge_gt)

                variables1_recurrent = torch.mean(torch.sigmoid(G_network_recurrent_bayesian1(val_img)[-1]) * torch.abs(val_edge_gt / M_recurrent  - (1 - val_edge_gt) / (1 - M_recurrent)))
                variables2_recurrent = torch.mean(torch.sigmoid(G_network_recurrent_bayesian2(val_img)[-1]) * torch.abs(val_edge_gt / M_recurrent  - (1 - val_edge_gt) / (1 - M_recurrent)))
                variables3_recurrent = torch.mean(torch.sigmoid(G_network_recurrent_bayesian3(val_img)[-1]) * torch.abs(val_edge_gt / M_recurrent  - (1 - val_edge_gt) / (1 - M_recurrent)))

                variables1_nonrecurrent = torch.mean(torch.sigmoid(G_network_nonrecurrent_bayesian1(val_img)[-1]) * torch.abs(val_edge_gt / M_nonrecurrent  - (1 - val_edge_gt) / (1 - M_nonrecurrent)))
                variables2_nonrecurrent = torch.mean(torch.sigmoid(G_network_nonrecurrent_bayesian2(val_img)[-1]) * torch.abs(val_edge_gt / M_nonrecurrent  - (1 - val_edge_gt) / (1 - M_nonrecurrent)))
                variables3_nonrecurrent = torch.mean(torch.sigmoid(G_network_nonrecurrent_bayesian3(val_img)[-1]) * torch.abs(val_edge_gt / M_nonrecurrent  - (1 - val_edge_gt) / (1 - M_nonrecurrent)))

                predictions_recurrent1 = predictions_recurrent1 + variables1_recurrent
                predictions_recurrent2 = predictions_recurrent2 + variables2_recurrent
                predictions_recurrent3 = predictions_recurrent3 + variables3_recurrent

                predictions_nonrecurrent1 = predictions_nonrecurrent1 + variables1_nonrecurrent
                predictions_nonrecurrent2 = predictions_nonrecurrent2 + variables2_nonrecurrent
                predictions_nonrecurrent3 = predictions_nonrecurrent3 + variables3_nonrecurrent
            
            constants = val_length * 0.5 / constants
            
            
            W1_recurrent = constants * predictions_recurrent1
            W2_recurrent = constants * predictions_recurrent2
            W3_recurrent = constants * predictions_recurrent3

            W1_nonrecurrent = constants * predictions_nonrecurrent1
            W2_nonrecurrent = constants * predictions_nonrecurrent2
            W3_nonrecurrent = constants * predictions_nonrecurrent3

            _W1_recurrent = W1_recurrent / (W1_recurrent + W2_recurrent + W3_recurrent)
            _W2_recurrent = W2_recurrent / (W1_recurrent + W2_recurrent + W3_recurrent)
            _W3_recurrent = W3_recurrent / (W1_recurrent + W2_recurrent + W3_recurrent)

            _W1_nonrecurrent = W1_nonrecurrent / (W1_nonrecurrent + W2_nonrecurrent + W3_nonrecurrent)
            _W2_nonrecurrent = W2_nonrecurrent / (W1_nonrecurrent + W2_nonrecurrent + W3_nonrecurrent)
            _W3_nonrecurrent = W3_nonrecurrent / (W1_nonrecurrent + W2_nonrecurrent + W3_nonrecurrent)

            W1_recurrent_previous = _W1_recurrent
            W2_recurrent_previous = _W2_recurrent
            W3_recurrent_previous = _W3_recurrent

            W1_nonrecurrent_previous = _W1_nonrecurrent
            W2_nonrecurrent_previous = _W2_nonrecurrent
            W3_nonrecurrent_previous = _W3_nonrecurrent

            
            print("Finish calculating weights of every parameter samplings")



        dis_weight = 0.8 * float(epoch) / float(args.n_epochs)

        loss_recurrent_meter = AverageMeters()
        loss_nonrecurrent_meter = AverageMeters()
        bar = tqdm.tqdm(dataloader, disable=True)
        saver.base_url = os.path.join(args.saved_path, 'results')

        for i, batch in enumerate(bar):
            # Set model input
            img = batch['img'].float().to(device)
            edge_gt = batch['edge'].float().to(device)

            if epoch >= 1:
                with torch.no_grad():
                    h, w = img.shape[2], img.shape[3]

                    #mask_features_recurrent_teacher    = G_network_teacher(img)[-1]
                    #mask_features_nonrecurrent    = G_network_teacher_nonrecurrent(img)[-1]
                    mask_features_recurrent_teacher = W1_recurrent_previous * G_network_recurrent_bayesian1(img)[-1] + W2_recurrent_previous * G_network_recurrent_bayesian2(img)[-1] + W3_recurrent_previous * G_network_recurrent_bayesian3(img)[-1]
                    mask_features_nonrecurrent_teacher = W1_nonrecurrent_previous * G_network_nonrecurrent_bayesian1(img)[-1] + W2_nonrecurrent_previous * G_network_nonrecurrent_bayesian2(img)[-1] + W3_nonrecurrent_previous * G_network_nonrecurrent_bayesian3(img)[-1]
 
                    uncertainty_recurrent = torch.abs(F.sigmoid(mask_features_recurrent_teacher) - 0.5).detach()
                    uncertainty_nonrecurrent = torch.abs(F.sigmoid(mask_features_nonrecurrent_teacher) - 0.5).detach()

                    weight = uncertainty_recurrent / (uncertainty_recurrent + uncertainty_nonrecurrent)

                    res = F.sigmoid(mask_features_recurrent_teacher * weight + mask_features_nonrecurrent_teacher * (1 - weight))
                    
                    edge_gt_soft = edge_gt * (1 - dis_weight) + res * dis_weight
            else:
                edge_gt_soft = edge_gt

            if random.random() < dis_weight:
                img_smoothed = bilateralFilter(img, 5)
                img = img_smoothed if random.random() > 0.5 else img + 2 * (img - img_smoothed)

            edge_feats_recurrent = G_network_recurrent(img)
            edge_preds_recurrent = [torch.sigmoid(r) for r in edge_feats_recurrent]

            # Identity loss
            loss_recurrent, loss_recurrent_items = criterion(edge_preds_recurrent, edge_gt, edge_gt_soft)

            if torch.isnan(loss):
                saver.save_image(img, './nan_im')
                saver.save_image(edge_gt, './nan_edge_gt')
                exit(0)
            loss_recurrent = loss_recurrent / args.iter_size
            loss_recurrent.backward()

            edge_feats_nonrecurrent = G_network_nonrecurrent(img)
            edge_preds_nonrecurrent = [torch.sigmoid(r) for r in edge_feats_nonrecurrent]

            # Identity loss
            loss_nonrecurrent, loss_nonrecurrent_items = criterion(edge_preds_nonrecurrent, edge_gt, edge_gt_soft)

            if torch.isnan(loss_nonrecurrent):
                saver.save_image(img, './nan_im')
                saver.save_image(edge_gt, './nan_edge_gt')
                exit(0)
            loss_nonrecurrent = loss_nonrecurrent / args.iter_size
            loss_nonrecurrent.backward()

            if (i + 1) % args.iter_size == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()

                optimizer_G_nonrecurrent.step()
                optimizer_G_nonrecurrent.zero_grad()

            loss_recurrent_meter.update(loss_recurrent_items)
            loss_nonrecurrent_meter.update(loss_nonrecurrent_items)

            if global_step % args.log_interval == 0:
                print('\r[Epoch %d/%d, Iter: %d/%d]: %s, %s' % (epoch, args.n_epochs, i, len(bar), loss_recurrent_meter, loss_nonrecurrent_meter), end="")
                write_loss(writer, 'train', loss_recurrent_meter, global_step)

            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    show = torch.cat([*edge_preds, edge_gt], dim=0).repeat(1, 3, 1, 1)
                    show = torch.cat([show, img], dim=0)
                    saver.save_image(show, '%09d' % global_step, nrow=5)

            global_step += 1

            del loss_recurrent, loss_nonrecurrent, img, edge_preds, edge_preds_nonrecurrent, edge_feats, edge_feats_nonrecurrent

        loss_recurrent_meter.reset()
        loss_nonrecurrent_meter.reset()
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            save_checkpoint({'G': G_network, 'G_teacher': G_network_teacher, 'G_nonrecurrent': G_network_nonrecurrent, 'G_teacher_nonrecurrent': G_network_teacher_nonrecurrent},
                            {'optimizer': optimizer_G, 'optimizer_nonrecurrent': optimizer_G_nonrecurrent},
                            {'scheduler': scheduler_cosine, 'scheduler_warmup': scheduler_warmup, 'scheduler_nonrecurrent': scheduler_cosine_nonrecurrent, 'scheduler_warmup_nonrecurrent': scheduler_warmup_nonrecurrent},
                            'ckt', epoch, os.path.join(args.saved_path, 'weights'))

        scheduler_warmup.step()
        scheduler_warmup_nonrecurrent.step()


if __name__ == '__main__':
    args = parser.parse_args()

    # setting random seed
    seed = 5603114
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = 'cuda'

    # Losses
    criterion = MyLoss().to(device)

    # Initialize student and teacher
    G_network_recurrent = Net_Recurrent().to(device)
    G_network_recurrent_teacher = Net_Recurrent().to(device)
    G_network_nonrecurrent = Net_NonRecurrent().to(device)
    G_network_nonrecurrent_teacher = Net_NonRecurrent().to(device)

    #Initialize Bayesian networks
    G_network_recurrent_bayesian1 = Net_Recurrent_bayesian().to(device)
    G_network_recurrent_bayesian2 = Net_Recurrent_bayesian().to(device)
    G_network_recurrent_bayesian3 = Net_Recurrent_bayesian().to(device)

    G_network_nonrecurrent_bayesian1 = Net_NonRecurrent_bayesian().to(device)
    G_network_nonrecurrent_bayesian2 = Net_NonRecurrent_bayesian().to(device)
    G_network_nonrecurrent_bayesian3 = Net_NonRecurrent_bayesian().to(device)


    # gradient stopping on momentum networks and Bayesian networks
    for p in G_network_recurrent_teacher.parameters():
        p.requires_grad = False
    for p in G_network_nonrecurrent_teacher.parameters():
        p.requires_grad = False

    for p in G_network_recurrent_bayesian1.parameters():
        p.requires_grad = False
    for p in G_network_recurrent_bayesian2.parameters():
        p.requires_grad = False
    for p in G_network_recurrent_bayesian3.parameters():
        p.requires_grad = False

    for p in G_network_nonrecurrent_bayesian1.parameters():
        p.requires_grad = False
    for p in G_network_nonrecurrent_bayesian2.parameters():
        p.requires_grad = False
    for p in G_network_nonrecurrent_bayesian3.parameters():
        p.requires_grad = False


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # Image transformations
    transforms_ = [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                   transforms.RandomGrayscale(p=0.2),                   
                   transforms.ToTensor(), normalize]

    # Training data loader
    dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, unaligned=True),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    # Testing data loader
    val_dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, unaligned=True, mode='val'),
                                batch_size=1, shuffle=False, num_workers=1)

    # Defining optimizer and schedulers

    optimizer_G_recurrent = torch.optim.AdamW(filter(lambda p: p.requires_grad, G_network_recurrent.parameters()),
                                    lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-3)
    scheduler_cosine_recurrent = CosineAnnealingLR(optimizer_G_recurrent, args.n_epochs)
    scheduler_warmup_recurrent = GradualWarmupScheduler(
        optimizer_G_recurrent, multiplier=8, total_epoch=4, after_scheduler=scheduler_cosine_recurrent)

    optimizer_G_nonrecurrent = torch.optim.AdamW(filter(lambda p: p.requires_grad, G_network_nonrecurrent.parameters()),
                                    lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-3)
    scheduler_cosine_nonrecurrent = CosineAnnealingLR(optimizer_G_nonrecurrent, args.n_epochs)
    scheduler_warmup_nonrecurrent = GradualWarmupScheduler(
        optimizer_G_nonrecurrent, multiplier=8, total_epoch=4, after_scheduler=scheduler_cosine_nonrecurrent)

    # Defining logging dirs
    timestamp = mutils.get_formatted_time()
    args.saved_path = args.saved_path + f'/{args.comment}/{timestamp}'
    args.log_path = args.log_path + f'/{args.comment}/{timestamp}/tensorboard/'

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.saved_path, exist_ok=True)

    writer = SingleSummaryWriter(args.log_path)
    global_step = 0
    W1_recurrent_previous = 0.33
    W2_recurrent_previous = 0.33
    W3_recurrent_previous = 0.33

    W1_nonrecurrent_previous = 0.33
    W2_nonrecurrent_previous = 0.33
    W3_nonrecurrent_previous = 0.33


    if args.resume is not None:
        state_dict = torch.load(args.resume)
        args.epoch = state_dict['epoch'] + 1
        G_network_recurrent.load_state_dict(state_dict['G_recurrent'])
        G_network_recurrent_teacher.load_state_dict(state_dict['G_recurrent_teacher'])
        G_network_nonrecurrent.load_state_dict(state_dict['G_nonrecurrent'])
        G_network_nonrecurrent_teacher.load_state_dict(state_dict['G_nonrecurrent_teacher'])
        optimizer_G_recurrent.load_state_dict(state_dict['optimizer_recurrent'])
        optimizer_G_nonrecurrent.load_state_dict(state_dict['optimizer_nonrecurrent'])
        scheduler_cosine_recurrent.load_state_dict(state_dict['scheduler_recurrent'])
        scheduler_warmup_recurrent.load_state_dict(state_dict['scheduler_warmup_recurrent'])
        scheduler_cosine_nonrecurrent.load_state_dict(state_dict['scheduler_nonrecurrent'])
        scheduler_warmup_nonrecurrent.load_state_dict(state_dict['scheduler_warmup_nonrecurrent'])

    main()
