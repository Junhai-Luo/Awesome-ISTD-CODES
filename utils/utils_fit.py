import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def _unpack_model_outputs(model_out):
    """Support both single-output and (outputs, aux_loss) model forwards."""
    if isinstance(model_out, (tuple, list)) and len(model_out) == 2:
        outputs, aux_loss = model_out
    else:
        outputs, aux_loss = model_out, 0.0
    return outputs, aux_loss


def _to_scalar_loss(loss_value):
    """Ensure loss is a scalar tensor for backward under DP/DDP."""
    if isinstance(loss_value, torch.Tensor) and loss_value.ndim > 0:
        return loss_value.mean()
    return loss_value


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    network_name = os.environ.get("NETWORK_NAME", "sstnet").lower()
    default_divisor = 1 if network_name == "sstnet" else 5
    epoch_step_divisor = max(1, int(os.environ.get("EPOCH_STEP_DIVISOR", default_divisor)))
    epoch_step = max(1, epoch_step // epoch_step_divisor)
    if local_rank == 0 and epoch_step_divisor != 1:
        print(f"Train epoch_step divisor: {epoch_step_divisor}")
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

        optimizer.zero_grad()
        if not fp16:
            model_out = model_train(images)
            outputs, motion_loss = _unpack_model_outputs(model_out)
            loss_value = yolo_loss(outputs, targets)
            if isinstance(motion_loss, torch.Tensor):
                loss_value = loss_value + motion_loss
            elif motion_loss:
                loss_value = loss_value + float(motion_loss)
            loss_value = _to_scalar_loss(loss_value)

            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                model_out = model_train(images)
                outputs, motion_loss = _unpack_model_outputs(model_out)
                loss_value = yolo_loss(outputs, targets)
                if isinstance(motion_loss, torch.Tensor):
                    loss_value = loss_value + motion_loss
                elif motion_loss:
                    loss_value = loss_value + float(motion_loss)
                loss_value = _to_scalar_loss(loss_value)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

            optimizer.zero_grad()
            model_out = model_train_eval(images)
            outputs, _ = _unpack_model_outputs(model_out)
            loss_value = yolo_loss(outputs, targets)
            loss_value = _to_scalar_loss(loss_value)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, 'best_epoch_weights.pth'))

        torch.save(save_state_dict, os.path.join(save_dir, 'last_epoch_weights.pth'))
