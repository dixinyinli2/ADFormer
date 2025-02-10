import torch
import os
import torch.nn.functional as F
import time
import numpy as np
from utils.utils import mae, rmse, mape
from utils.utils import WarmupCosineAnnealingLR
import matplotlib.pyplot as plt



class ADFormerTrainer():
    def __init__(self, args, model, logger):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.logger = logger
        self.scaler = model.scaler

    def train(self, train_dataloader, val_dataloader):
        self.logger.info("Start training ...... ")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                           eps=self.args.lr_epsilon, betas=(self.args.lr_beta1, self.args.lr_beta2), weight_decay=self.args.weight_decay)
        
        if self.args.lr_decay:
            if self.args.lr_scheduler == 'WarmupCosineAnnealingLR':
                self.lr_scheduler = WarmupCosineAnnealingLR(self.optimizer, T_max=self.args.lr_T_max, warmup_t=self.args.lr_warmup_epoch,
                                                            warmup_lr_init=self.args.lr_warmup_init, eta_min=self.args.lr_eta_min)
                
        train_losses, val_losses = [], []
        min_val_loss = float('inf')
        batch_seen = 0
        for epoch_idx in range(self.args.epochs):
            t1 = time.time()
            train_loss, batch_seen = self.train_epoch(train_dataloader, batch_seen)
            t2 = time.time()
            val_loss = self.val_epoch(val_dataloader)
            t3 = time.time()

            if epoch_idx >= 4:
                train_losses.append(train_loss)
                val_losses.append(val_loss)

            if self.args.lr_decay:
                if self.args.lr_scheduler == 'WarmupCosineAnnealingLR':
                    self.lr_scheduler.step()

            log_lr = self.optimizer.param_groups[0]['lr']
            message = 'Epoch {}, train loss: {:.4f}; val loss: {:.4f}; lr: {:.6f}, t1: {:.2f}s, t2: {:.2f}s'. \
                format(epoch_idx, train_loss, val_loss, log_lr, t2-t1, t3-t2)
            self.logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                self.logger.info(f'Val loss decreases from {min_val_loss: .4f} to {val_loss: .4f}')
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.args.log_dir, 'model_best_state.pth'))
            else:
                wait += 1
                if self.args.use_early_stop and wait == self.args.patience:
                    self.logger.info(f"No improvement with {wait} epoches, training stop at epoch {epoch_idx}.")
                    break

        self.logger.info('Training stop ...... start to evaluate')
        self.plot_loss(train_losses=train_losses, val_losses=val_losses)

    
    def train_epoch(self, dataloader, batch_seen):
        self.model.train()
        losses = []

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.l1_loss(self.scaler.inverse_transform(y[..., :self.args.output_dim]), self.scaler.inverse_transform(output))
            losses.append(loss.item())
            batch_seen += 1
            loss.backward()

            if self.args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return np.mean(losses), batch_seen 


    def val_epoch(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            losses = []
            
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = F.l1_loss(self.scaler.inverse_transform(y[..., :self.args.output_dim]), self.scaler.inverse_transform(output))
                losses.append(loss.item())

            return np.mean(losses)
        
    def plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        epochs = range(5, 5 + len(train_losses))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.args.log_dir, 'loss_curve.png'))


    def evaluate(self, test_dataloader):

        with torch.no_grad():
            self.model.eval()
            real_y = []
            pred_y = []

            for x, y in test_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                y = self.scaler.inverse_transform(y[..., :self.args.output_dim])
                y_hat = self.scaler.inverse_transform(output)
                real_y.append(y.detach().cpu().numpy())
                pred_y.append(y_hat.detach().cpu().numpy())
            real_y = np.concatenate(real_y, axis=0)
            pred_y = np.concatenate(pred_y, axis=0)

            outputs = {'prediction': pred_y, 'truth': real_y}
            np.savez_compressed(os.path.join(self.args.log_dir, 'predictions.npz'), **outputs)

            MAE = mae(real_y, pred_y, mask=self.args.metric_mask)
            RMSE = rmse(real_y, pred_y, mask=self.args.metric_mask)
            MAPE= mape(real_y, pred_y)

            self.logger.info(f'prediction on {real_y.shape[1]} steps, MAE: {MAE: .4f}, RMSE: {RMSE: .4f}, MAPE: {MAPE: .4f}')
            return MAE, RMSE, MAPE

            


