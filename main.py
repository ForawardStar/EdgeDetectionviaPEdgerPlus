import argparse

import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import *
from loss_function import MyLoss
from models import *
from models_noshare import Guider_noshare
from models_bayesian import *
from models_noshare_bayesian import Guider_noshare_bayesian
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
    global W1_previous
    global W2_previous
    global W3_previous

    global W1_noshare_previous
    global W2_noshare_previous
    global W3_noshare_previous


    for epoch in range(args.epoch, args.n_epochs):
        if epoch >= 2:
            state_st = G_network.state_dict()
            state_t = G_network_teacher.state_dict()
            for k, v in state_t.items():
                state_t[k] = (state_t[k] + state_st[k]) * 0.5
            G_network_teacher.load_state_dict(state_t)

            state_st_noshare = G_network_noshare.state_dict()
            state_t_noshare = G_network_teacher_noshare.state_dict()
            for k, v in state_t_noshare.items():
                state_t_noshare[k] = (state_t_noshare[k] + state_st_noshare[k]) * 0.5
            G_network_teacher_noshare.load_state_dict(state_t_noshare)

        elif epoch == 1:
            G_network_teacher.load_state_dict(G_network.state_dict())

            G_network_teacher_noshare.load_state_dict(G_network_noshare.state_dict())

        if epoch >= 1:
            # Update the parameters of Bayesian Networks
            state_t = G_network_teacher.state_dict()

            state_b1 = G_network_bayesian1.state_dict()
            for k, v in state_t.items():
                state_b1[k] = state_t[k]
            G_network_bayesian1.load_state_dict(state_b1)

            state_b2 = G_network_bayesian2.state_dict()
            for k, v in state_t.items():
                state_b2[k] = state_t[k]
            G_network_bayesian2.load_state_dict(state_b2)

            state_b3 = G_network_bayesian3.state_dict()
            for k, v in state_t.items():
                state_b3[k] = state_t[k]
            G_network_bayesian3.load_state_dict(state_b3)

            ##################################################################333
            state_t_noshare = G_network_teacher_noshare.state_dict()

            state_b1_noshare = G_network_noshare_bayesian1.state_dict()
            for k, v in state_t_noshare.items():
                state_b1_noshare[k] = state_t_noshare[k]
            G_network_noshare_bayesian1.load_state_dict(state_b1_noshare)

            state_b2_noshare = G_network_noshare_bayesian2.state_dict()
            for k, v in state_t_noshare.items():
                state_b2_noshare[k] = state_t_noshare[k]
            G_network_noshare_bayesian2.load_state_dict(state_b2_noshare)

            state_b3_noshare = G_network_noshare_bayesian3.state_dict()
            for k, v in state_t_noshare.items():
                state_b3_noshare[k] = state_t_noshare[k]
            G_network_noshare_bayesian3.load_state_dict(state_b3_noshare)
      
           
            print("Finish sampling parameters")
            # Computing weights of every parameter samplings 
            bar_val = tqdm.tqdm(val_dataloader, disable=True)
            val_length = len(bar_val)
            constants = 0.0
            predictions1 = 0.0
            predictions2 = 0.0
            predictions3 = 0.0
            predictions_noshare1 = 0.0
            predictions_noshare2 = 0.0
            predictions_noshare3 = 0.0
            for i, val_batch in enumerate(bar_val):
                val_img = val_batch['img'].float().to(device)
                val_edge_gt = val_batch['edge'].float().to(device)

                M = W1_previous * torch.sigmoid(G_network_bayesian1(val_img)[-1]) + W2_previous * torch.sigmoid(G_network_bayesian2(val_img)[-1]) + W3_previous * torch.sigmoid(G_network_bayesian3(val_img)[-1])
                M_noshare = W1_noshare_previous * torch.sigmoid(G_network_noshare_bayesian1(val_img)[-1]) + W2_noshare_previous * torch.sigmoid(G_network_noshare_bayesian2(val_img)[-1]) + W3_noshare_previous * torch.sigmoid(G_network_noshare_bayesian3(val_img)[-1])

                constants = constants + torch.mean(val_edge_gt)

                variables1 = torch.mean(torch.sigmoid(G_network_bayesian1(val_img)[-1]) * torch.abs(val_edge_gt / M  - (1 - val_edge_gt) / (1 - M)))
                variables2 = torch.mean(torch.sigmoid(G_network_bayesian2(val_img)[-1]) * torch.abs(val_edge_gt / M  - (1 - val_edge_gt) / (1 - M)))
                variables3 = torch.mean(torch.sigmoid(G_network_bayesian3(val_img)[-1]) * torch.abs(val_edge_gt / M  - (1 - val_edge_gt) / (1 - M)))

                variables1_noshare = torch.mean(torch.sigmoid(G_network_noshare_bayesian1(val_img)[-1]) * torch.abs(val_edge_gt / M_noshare  - (1 - val_edge_gt) / (1 - M_noshare)))
                variables2_noshare = torch.mean(torch.sigmoid(G_network_noshare_bayesian2(val_img)[-1]) * torch.abs(val_edge_gt / M_noshare  - (1 - val_edge_gt) / (1 - M_noshare)))
                variables3_noshare = torch.mean(torch.sigmoid(G_network_noshare_bayesian3(val_img)[-1]) * torch.abs(val_edge_gt / M_noshare  - (1 - val_edge_gt) / (1 - M_noshare)))

                predictions1 = predictions1 + variables1
                predictions2 = predictions2 + variables2
                predictions3 = predictions3 + variables3

                predictions_noshare1 = predictions1 + variables1_noshare
                predictions_noshare2 = predictions2 + variables2_noshare
                predictions_noshare3 = predictions3 + variables3_noshare
            
            constants = val_length * 0.5 / constants
            
            W1_previous = W1
            W2_previous = W2
            W3_previous = W3

            W1_noshare_previous = W1_noshare
            W2_noshare_previous = W2_noshare
            W3_noshare_previous = W3_noshare

            W1 = constants * predictions1
            W2 = constants * predictions2
            W3 = constants * predictions3

            W1_noshare = constants * predictions_noshare1
            W2_noshare = constants * predictions_noshare2
            W3_noshare = constants * predictions_noshare3

            _W1 = W1 / (W1 + W2 + W3)
            _W2 = W2 / (W1 + W2 + W3)
            _W3 = W3 / (W1 + W2 + W3)

            _W1_noshare = W1_noshare / (W1_noshare + W2_noshare + W3_noshare)
            _W2_noshare = W2_noshare / (W1_noshare + W2_noshare + W3_noshare)
            _W3_noshare = W3_noshare / (W1_noshare + W2_noshare + W3_noshare)

            print("Finish calculating weights of every parameter samplings. _W1:{}  ,  _W2:{}  ,  _W3:{}  ,  _W1_noshare:{}  ,  _W2_noshare:{}  ,  _W3_noshare:{}".format(_W1,_W2,_W3,_W1_noshare,_W2_noshare,_W3_noshare))



        dis_weight = 0.8 * float(epoch) / float(args.n_epochs)

        loss_meter = AverageMeters()
        loss_noshare_meter = AverageMeters()
        bar = tqdm.tqdm(dataloader, disable=True)
        saver.base_url = os.path.join(args.saved_path, 'results')

        for i, batch in enumerate(bar):
            # if args.debug and i > 2000:
            #     break

            # Set model input
            img = batch['img'].float().to(device)
            edge_gt = batch['edge'].float().to(device)

            if epoch >= 1:
                with torch.no_grad():
                    h, w = img.shape[2], img.shape[3]

                    #mask_features_teacher    = G_network_teacher(img)[-1]
                    #mask_features_noshare    = G_network_teacher_noshare(img)[-1]
                    mask_features_teacher = W1_previous * G_network_bayesian1(val_img)[-1] + W2_previous * G_network_bayesian2(val_img)[-1] + W3_previous * G_network_bayesian3(val_img)[-1]
                    mask_features_noshare_teacher = W1_noshare_previous * G_network_noshare_bayesian1(val_img)[-1] + W2_noshare_previous * G_network_noshare_bayesian2(val_img)[-1] + W3_noshare_previous * G_network_noshare_bayesian3(val_img)[-1]
 
                    uncertainty = torch.abs(F.sigmoid(mask_features_teacher) - 0.5).detach()
                    uncertainty_noshare = torch.abs(F.sigmoid(mask_features_noshare_teacher) - 0.5).detach()

                    weight = uncertainty / (uncertainty + uncertainty_noshare)

                    res = F.sigmoid(mask_features_teacher * weight + mask_features_noshare_teacher * (1 - weight))
                    
                    edge_gt_soft = edge_gt * (1 - dis_weight) + res * dis_weight
            else:
                edge_gt_soft = edge_gt

            if random.random() < dis_weight:
                img_smoothed = bilateralFilter(img, 5)
                img = img_smoothed if random.random() > 0.5 else img + 2 * (img - img_smoothed)

            edge_feats = G_network(img)
            edge_preds = [torch.sigmoid(r) for r in edge_feats]

            # Identity loss
            loss, loss_items = criterion(edge_preds, edge_gt, edge_gt_soft)

            if torch.isnan(loss):
                saver.save_image(img, './nan_im')
                saver.save_image(edge_gt, './nan_edge_gt')
                exit(0)
            loss = loss / args.iter_size
            loss.backward()

            edge_feats_noshare = G_network_noshare(img)
            edge_preds_noshare = [torch.sigmoid(r) for r in edge_feats_noshare]

            # Identity loss
            loss_noshare, loss_noshare_items = criterion(edge_preds_noshare, edge_gt, edge_gt_soft)

            if torch.isnan(loss_noshare):
                saver.save_image(img, './nan_im')
                saver.save_image(edge_gt, './nan_edge_gt')
                exit(0)
            loss_noshare = loss_noshare / args.iter_size
            loss_noshare.backward()

            if (i + 1) % args.iter_size == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()

                optimizer_G_noshare.step()
                optimizer_G_noshare.zero_grad()

            loss_meter.update(loss_items)
            loss_noshare_meter.update(loss_noshare_items)

            if global_step % args.log_interval == 0:
                print('\r[Epoch %d/%d, Iter: %d/%d]: %s, %s' % (epoch, args.n_epochs, i, len(bar), loss_meter, loss_noshare_meter), end="")
                write_loss(writer, 'train', loss_meter, global_step)

            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    show = torch.cat([*edge_preds, edge_gt], dim=0).repeat(1, 3, 1, 1)
                    show = torch.cat([show, img], dim=0)
                    saver.save_image(show, '%09d' % global_step, nrow=5)

            global_step += 1

            del loss, loss_noshare, img, edge_preds, edge_preds_noshare, edge_feats, edge_feats_noshare

        loss_meter.reset()
        loss_noshare_meter.reset()
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            save_checkpoint({'G': G_network, 'G_teacher': G_network_teacher, 'G_noshare': G_network_noshare, 'G_teacher_noshare': G_network_teacher_noshare},
                            {'optimizer': optimizer_G, 'optimizer_noshare': optimizer_G_noshare},
                            {'scheduler': scheduler_cosine, 'scheduler_warmup': scheduler_warmup, 'scheduler_noshare': scheduler_cosine_noshare, 'scheduler_warmup_noshare': scheduler_warmup_noshare},
                            'ckt', epoch, os.path.join(args.saved_path, 'weights'))

        scheduler_warmup.step()
        scheduler_warmup_noshare.step()


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
    G_network = Guider_stu().to(device)
    G_network_teacher = Guider_stu().to(device)
    G_network_noshare = Guider_noshare().to(device)
    G_network_teacher_noshare = Guider_noshare().to(device)

    #Initialize Bayesian networks
    G_network_bayesian1 = Guider_stu_bayesian().to(device)
    G_network_bayesian2 = Guider_stu_bayesian().to(device)
    G_network_bayesian3 = Guider_stu_bayesian().to(device)

    G_network_noshare_bayesian1 = Guider_noshare_bayesian().to(device)
    G_network_noshare_bayesian2 = Guider_noshare_bayesian().to(device)
    G_network_noshare_bayesian3 = Guider_noshare_bayesian().to(device)


    # gradient stopping on momentum networks and Bayesian networks
    for p in G_network_teacher.parameters():
        p.requires_grad = False
    for p in G_network_teacher_noshare.parameters():
        p.requires_grad = False

    for p in G_network_bayesian1.parameters():
        p.requires_grad = False
    for p in G_network_bayesian2.parameters():
        p.requires_grad = False
    for p in G_network_bayesian3.parameters():
        p.requires_grad = False

    for p in G_network_noshare_bayesian1.parameters():
        p.requires_grad = False
    for p in G_network_noshare_bayesian2.parameters():
        p.requires_grad = False
    for p in G_network_noshare_bayesian3.parameters():
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

    optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, G_network.parameters()),
                                    lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-3)
    scheduler_cosine = CosineAnnealingLR(optimizer_G, args.n_epochs)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer_G, multiplier=8, total_epoch=4, after_scheduler=scheduler_cosine)

    optimizer_G_noshare = torch.optim.AdamW(filter(lambda p: p.requires_grad, G_network_noshare.parameters()),
                                    lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-3)
    scheduler_cosine_noshare = CosineAnnealingLR(optimizer_G_noshare, args.n_epochs)
    scheduler_warmup_noshare = GradualWarmupScheduler(
        optimizer_G_noshare, multiplier=8, total_epoch=4, after_scheduler=scheduler_cosine_noshare)

    # Defining logging dirs
    timestamp = mutils.get_formatted_time()
    args.saved_path = args.saved_path + f'/{args.comment}/{timestamp}'
    args.log_path = args.log_path + f'/{args.comment}/{timestamp}/tensorboard/'

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.saved_path, exist_ok=True)

    writer = SingleSummaryWriter(args.log_path)
    global_step = 0
    W1_previous = 0.33
    W2_previous = 0.33
    W3_previous = 0.33

    W1_noshare_previous = 0.33
    W2_noshare_previous = 0.33
    W3_noshare_previous = 0.33


    if args.resume is not None:
        state_dict = torch.load(args.resume)
        args.epoch = state_dict['epoch'] + 1
        G_network.load_state_dict(state_dict['G'])
        G_network_teacher.load_state_dict(state_dict['G_teacher'])
        G_network_noshare.load_state_dict(state_dict['G_noshare'])
        G_network_teacher_noshare.load_state_dict(state_dict['G_teacher_noshare'])
        optimizer_G.load_state_dict(state_dict['optimizer'])
        optimizer_G_noshare.load_state_dict(state_dict['optimizer_noshare'])
        scheduler_cosine.load_state_dict(state_dict['scheduler'])
        scheduler_warmup.load_state_dict(state_dict['scheduler_warmup'])
        scheduler_cosine_noshare.load_state_dict(state_dict['scheduler_noshare'])
        scheduler_warmup_noshare.load_state_dict(state_dict['scheduler_warmup_noshare'])

    main()
