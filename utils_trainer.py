# package import
import os
from apex.parallel import DistributedDataParallel as DDP
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
# import wandb

# from sklearn.metrics import roc_auc_score

from tqdm import  tqdm

import numpy as np
import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from pathlib import Path
import time
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# image-text embedding diagnosis style trainer Class (with language model)
class trainer_wBert:
    def __init__(self, model,
                 optimizer, device, model_name, args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.loss_type = args.loss_type
        self.train_batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.max_epochs = args.max_epochs
        self.lr_max = args.lr
        self.num_workers = args.num_workers
        # self.checkpoint_interval = args['checkpoint_interval']
        self.smooth = args.smooth
        self.prior_ratio = args.ratio

        self.args = args

        self.training_data = None

    def covar_loss(self, img_embed, text_embed):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        logits = torch.mm(img_embed.T, text_embed).to(self.device)

        logits.div_(self.train_batch_size)
        on_diag = torch.diagonal(logits).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(logits).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss / 2.

    def reg_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        loss += 2 - 2 * (y * x).sum(dim=-1)
        return loss.mean()

    def entropy_loss(self, x, y):
        x = F.log_softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)
        metric = nn.KLDivLoss(reduction="batchmean")
        loss = metric(x, y)
        return loss.mean()


    def my_collate(self, batch):
        len_batch = len(batch)
        batch = list(filter(lambda x: x is not None, batch))
        if len_batch > len(batch):
            diff = len_batch - len(batch)
            for i in range(diff):
                item = self.training_data[np.random.randint(0, len(self.training_data))]
                while item is None:
                    item = self.training_data[np.random.randint(0, len(self.training_data))]
                batch.append(item)
        return torch.utils.data.dataloader.default_collate(batch)



    # traing process
    def train_w_TextEmb(self, train_dataset, bash_save_dir, bash_tsbd_dir):

        # using the tensordboard
        self.training_data = train_dataset

        if self.args.rank == 0:
            tb_writer = SummaryWriter(bash_tsbd_dir)

        train_loader = DataLoader(self.training_data, batch_size=self.train_batch_size, num_workers=8,
                                  drop_last=True, shuffle=False,
                                  sampler=DistributedSampler(train_dataset, num_replicas=1, rank=0))

        model_checkpoints_folder = bash_save_dir

        if not os.path.exists(model_checkpoints_folder):
            print('create directory "{}" for save checkpoint!'.format(model_checkpoints_folder))
            print('---------------------------')
            os.mkdir(model_checkpoints_folder)
        else:
            print('directory "{}" existing for save checkpoint!'.format(model_checkpoints_folder))

        # automatically resume from checkpoint if it exists
        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        if os.path.exists(model_checkpoints_folder + self.model_name + '_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name + '_checkpoint.pth', map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
        else:
            start_epoch = 0
            print('Start training from 0 epoch')

        print('#########################################')
        print('training start!')

        # cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(self.max_epochs * len(train_dataset) //
                    self.args.world_size // self.train_batch_size * 0.4),
            T_mult=1,
            eta_min=1e-8,
        )

        niter = 1

        skip_scheduler = False

        # setting loss step:
        global_step = 0

        tr_clip_loss = 0
        tr_slip_loss = 0
        tr_lambda_t_loss = 0
        tr_text_i_loss = 0
        tr_text_f_loss = 0

        logging_clip_loss = 0
        logging_slip_loss = 0
        logging_lambda_t_loss = 0
        logging_tr_text_i_loss = 0
        logging_tr_text_f_loss = 0

        end_time, start_time = 0, 0
        step_in_each_epoch = len(train_loader) // self.args.gradient_accumulation_steps

        ############### prepare optimizer and amp training #######################

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                  " to use fp16 training.")

            model, optimizer = amp.initialize(self.model, self.optimizer,
                                              opt_level=self.args.fp16_opt_level,
                                              min_loss_scale=1)  #
            self.model = model
            self.optimizer = optimizer

            print('Start to using FP 16 ###################################')

        self.model = DDP(self.model, message_size=10000000,
                         gradient_predivide_factor=torch.distributed.get_world_size(),
                         delay_allreduce=True)
        logger.info('apex data paralleled!')

        for name, par in self.model.module.lm_model.named_parameters():
            print('name and par', name, par.requires_grad)

        self.model.train()
        for epoch in tqdm(range(start_epoch, self.max_epochs + 1)):

            epoch_loss = 0

            for step, data in enumerate(train_loader):

                step += 1

                text = data['raw_text']['IMP']

                img_1 = data['image1'].to(torch.float32).to(self.device).contiguous()
                img_2 = data['image2'].to(torch.float32).to(self.device).contiguous()

                text_domain_label = data['text_label'].to(torch.float32).unsqueeze(-1)

                imp_tokenize_output = self.model.module._tokenize(text)

                input_ids, attention_mask = {}, {}
                input_ids['imp'] = imp_tokenize_output.input_ids.to(self.device).contiguous()
                attention_mask['imp'] = imp_tokenize_output.attention_mask.to(self.device).contiguous()

                if self.loss_type == 'clip_loss':

                    output_dict = self.model(img_1, img_2, text_domain_label, input_ids['imp'], attention_mask['imp'], unified_train=False, text_aug=0)
                    clip_loss = output_dict['clip_loss']
                    ti_acc1 = output_dict['ti_acc1'].item()
                    ti_acc5 = output_dict['ti_acc5'].item()

                    loss = clip_loss
                    # accumalate loss for logging
                    epoch_loss += loss.item()

                    tr_clip_loss += loss.item()

                elif self.loss_type == 'unified_loss':

                    output_dict = self.model(img_1, img_2, text_domain_label, input_ids['imp'], attention_mask['imp'], unified_train=True, text_aug=0)
                    clip_loss = output_dict['clip_loss']
                    slip_loss = output_dict['slip_loss']
                    text_d_loss = output_dict['text_d_loss']
                    ti_acc1 = output_dict['ti_acc1'].item()
                    ti_acc5 = output_dict['ti_acc5'].item()
                    ii_acc1 = output_dict['ii_acc1'].item()
                    ii_acc5 = output_dict['ii_acc5'].item()

                    text_d_loss = self.args.lambda_t * text_d_loss
                    loss = clip_loss + slip_loss + text_d_loss
                    # accumalate loss for logging
                    epoch_loss += loss.item()

                    tr_clip_loss += clip_loss.item()
                    tr_slip_loss += slip_loss.item()
                    tr_lambda_t_loss += text_d_loss.item()

                elif self.loss_type == 'unified_loss_total':

                    output_dict = self.model(img_1, img_2, text_domain_label, input_ids['imp'], attention_mask['imp'],
                                             unified_train=True, text_aug=self.args.text_aug)
                    clip_loss = output_dict['clip_loss']
                    slip_loss = output_dict['slip_loss']
                    text_d_loss = output_dict['text_d_loss']

                    text_instance_loss = output_dict['text_instance_loss']
                    text_feature_loss = output_dict['text_feature_loss']
                    ti_acc1 = output_dict['ti_acc1'].item()
                    ti_acc5 = output_dict['ti_acc5'].item()
                    ii_acc1 = output_dict['ii_acc1'].item()
                    ii_acc5 = output_dict['ii_acc5'].item()

                    text_d_loss = self.args.lambda_t * text_d_loss
                    loss = clip_loss + slip_loss + text_d_loss
                    # accumalate loss for logging
                    epoch_loss += loss.item()

                    tr_clip_loss += clip_loss.item()
                    tr_slip_loss += slip_loss.item()
                    tr_lambda_t_loss += text_d_loss.item()
                    tr_text_i_loss += text_instance_loss.item()
                    tr_text_f_loss += text_feature_loss.item()


                if self.args.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)

                # using gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if (step + 1) % (self.args.gradient_accumulation_steps * self.args.logging_steps) == 0 \
                            and self.args.local_rank < 2 or global_step < 5000:
                        end_time = time.time()

                        if self.loss_type == 'clip_loss':

                            logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s; acc1: %s, acc5: %s'
                                ' (%.2f sec)' %
                                (epoch, global_step + 1, step_in_each_epoch, self.optimizer.param_groups[0]['lr'],
                                 loss.item() * self.args.gradient_accumulation_steps,ti_acc1, ti_acc5,
                                 end_time - start_time))

                            print('The current step loss is : ', loss.item())

                        elif self.loss_type == 'unified_loss':

                            logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s, text_d_loss is %s; clip_acc1: %s, clip_acc5: %s; slip_acc1: %s, slip_acc5: %s'
                                ' (%.2f sec)' %
                                (epoch, global_step + 1, step_in_each_epoch, self.optimizer.param_groups[0]['lr'],
                                 loss.item() * self.args.gradient_accumulation_steps, text_d_loss.item() * self.args.gradient_accumulation_steps ,ti_acc1, ti_acc5, ii_acc1, ii_acc5,
                                 end_time - start_time))

                            print('The current step loss is : ', loss.item())

                        elif self.loss_type == 'unified_loss_total':
                            logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s, text_d_loss is %s, text_i_loss % s, text_f_loss % s; clip_acc1: %s, clip_acc5: %s; slip_acc1: %s, slip_acc5: %s'
                                ' (%.2f sec)' %
                                (epoch, global_step + 1, step_in_each_epoch, self.optimizer.param_groups[0]['lr'],
                                 loss.item() * self.args.gradient_accumulation_steps,
                                 text_d_loss.item() * self.args.gradient_accumulation_steps,
                                 text_instance_loss.item() * self.args.gradient_accumulation_steps,
                                 text_feature_loss.item() * self.args.gradient_accumulation_steps,
                                 ti_acc1, ti_acc5, ii_acc1,
                                 ii_acc5,
                                 end_time - start_time))

                            print('The current step loss is : ', loss.item())

                        start_time = time.time()

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0 and self.args.rank == 0:

                        if self.loss_type == 'clip_loss':

                            tb_writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], global_step)
                            tb_writer.add_scalar("step_clip_loss", (
                                    tr_clip_loss - logging_clip_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)
                            tb_writer.add_scalar("clip_acc1", ti_acc1, global_step)
                            tb_writer.add_scalar("clip_acc5", ti_acc5, global_step)

                            logging_clip_loss = tr_clip_loss

                        elif self.loss_type == 'unified_loss':

                            tb_writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], global_step)
                            tb_writer.add_scalar("step_clip_loss", (
                                    tr_clip_loss - logging_clip_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)

                            tb_writer.add_scalar("step_slip_loss", (
                                    tr_slip_loss - logging_slip_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)

                            tb_writer.add_scalar("logging_lambda_t_loss", (
                                    tr_lambda_t_loss - logging_lambda_t_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)

                            tb_writer.add_scalar("clip_acc1", ti_acc1, global_step)
                            tb_writer.add_scalar("clip_acc5", ti_acc5, global_step)
                            tb_writer.add_scalar("slip_acc1", ii_acc1, global_step)
                            tb_writer.add_scalar("slip_acc5", ii_acc5, global_step)


                            logging_clip_loss = tr_clip_loss
                            logging_slip_loss = tr_slip_loss
                            logging_lambda_t_loss = tr_lambda_t_loss

                        elif self.loss_type == 'unified_loss_total':

                            tb_writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], global_step)
                            tb_writer.add_scalar("step_clip_loss", (
                                    tr_clip_loss - logging_clip_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)

                            tb_writer.add_scalar("step_slip_loss", (
                                    tr_slip_loss - logging_slip_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)

                            tb_writer.add_scalar("text_domain_loss", (
                                    tr_lambda_t_loss - logging_lambda_t_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)

                            tb_writer.add_scalar("text_instance_loss", (
                                    tr_text_i_loss - logging_tr_text_i_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)
                            tb_writer.add_scalar("text_feature_loss", (
                                    tr_text_f_loss - logging_tr_text_f_loss) / self.args.logging_steps / self.args.gradient_accumulation_steps,
                                                 global_step)

                            tb_writer.add_scalar("clip_acc1", ti_acc1, global_step)
                            tb_writer.add_scalar("clip_acc5", ti_acc5, global_step)
                            tb_writer.add_scalar("slip_acc1", ii_acc1, global_step)
                            tb_writer.add_scalar("slip_acc5", ii_acc5, global_step)

                            logging_clip_loss = tr_clip_loss
                            logging_slip_loss = tr_slip_loss
                            logging_lambda_t_loss = tr_lambda_t_loss
                            logging_tr_text_i_loss = tr_text_i_loss
                            logging_tr_text_f_loss = tr_text_f_loss


                niter += 1
                if not skip_scheduler:
                    scheduler.step()

            if self.args.rank == 0:
                epoch_iter = (len(train_dataset)//self.train_batch_size//self.args.world_size)
                print(f'{epoch} epoch loss is {epoch_loss / epoch_iter}!')
                tb_writer.add_scalar("epoch_avg_loss", epoch_loss / epoch_iter, epoch)


            if self.args.rank == 0 and epoch % 5 == 0:
                saving_path = bash_save_dir
                saving_path = Path(os.path.join(saving_path, "epoch_" + str(epoch)))

                if saving_path.is_dir() and list(saving_path.iterdir()):
                    logging.warning(f"Output directory ({saving_path}) already exists and is not empty!")
                saving_path.mkdir(parents=True, exist_ok=True)
                logging.info("** ** * Saving fine-tuned vision encoder model ** ** * ")

                torch.save(self.model.module.encoder.state_dict(),
                           Path(os.path.join(saving_path, self.model_name + '_encoder.pth')))

                if self.args.freeze_layers != 12:
                    logging.info("** ** * Saving fine-tuned mmodal model ** ** * ")
                    torch.save(self.model.module.state_dict(),
                               Path(os.path.join(saving_path, self.model_name + '_total.pth')))

        ############################################# FINAL EPOCH SAVING ###############################################

        # save the gradient-update vision encoder
        torch.save(self.model.module.encoder.state_dict(),
                   Path(os.path.join(model_checkpoints_folder, '_final_' + self.model_name + '_encoder.pth')))

        # save final total model
        if self.args.freeze_layers != 12:
            # save the gradient update text encoder
            self.model = self.model.eval()


            final_saving_path = Path(os.path.join(bash_save_dir, "final"))

            if final_saving_path.is_dir() and list(final_saving_path.iterdir()):
                logging.warning(f"Output directory ({final_saving_path}) already exists and is not empty!")
            final_saving_path.mkdir(parents=True, exist_ok=True)

            logging.info("** ** * Saving fine-tuned model ** ** * ")
            model_to_save = self.model.module.lm_model  # Only save the model it-self

            CONFIG_NAME = "config.json"
            WEIGHTS_NAME = "pytorch_model.bin"
            output_model_file = os.path.join(final_saving_path, WEIGHTS_NAME)
            output_config_file = os.path.join(final_saving_path, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            model_to_save.tokenizer.save_vocabulary(final_saving_path)

            print('Save text encoder failure....')
            # save the final total model
            logging.info("** ** * Saving fine-tuned mmodal model ok ** ** * ")
            torch.save(self.model.module.state_dict(),
                       model_checkpoints_folder + '_final_' + self.model_name + 'total_.pth')

    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)
