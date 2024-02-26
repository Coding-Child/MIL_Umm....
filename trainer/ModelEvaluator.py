import os
from util.metric import calculate_metrics
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from util.remove_file import clear_directory


def evaluate_model(model, data_loader, criterion_1, criterion_2, fold=None, epoch=None, loss_weight=None, phase='Validation'):
    """
    params:
        model: nn.Module (model)
        criterion_1: nn.Module (instance loss)
        criterion_2: nn.Module (bag loss)
        data_loader: DataLoader (data loader)
    """
    total_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0
    total_samples = 0
    y_true = []
    y_pred = []

    model.eval()
    with tqdm(data_loader, desc=f'{phase}', unit='b', ascii=True, ncols=150) as pbar:
        with torch.no_grad():
            for _, inputs, targets, pseudo_targets in data_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                pseudo_targets = pseudo_targets.cuda()

                instance_pred, bag_pred, _ = model(inputs)
                loss_1 = criterion_1(instance_pred, pseudo_targets.view(-1))
                loss_2 = criterion_2(bag_pred, targets)

                if phase == 'Validation':
                    loss = (loss_1 * loss_weight) + (loss_2 * (1 - loss_weight))
                    total_loss += loss.item() * inputs.size(0)

                total_loss_1 += loss_1.item() * inputs.size(0)
                total_loss_2 += loss_2.item() * inputs.size(0)
                total_samples += inputs.size(0)

                y_true.extend(targets.detach().cpu().numpy())
                y_pred.extend(bag_pred.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_postfix_str(f'instance: {loss_1.item():.4f}, bag: {loss_2.item():.4f}')

                torch.cuda.empty_cache()
                del inputs, targets, pseudo_targets

    auroc, f1_score, acc, precision, recall = calculate_metrics(y_true, y_pred)

    if phase == 'Final Test':
        os.makedirs(f'test_result/fold_{fold + 1}', exist_ok=True)

        result_dict = {'target': y_true, 'pred': y_pred}

        df = pd.DataFrame(result_dict)
        df.to_csv(f'test_result/fold_{fold + 1}/final_test_fold_{fold + 1}.csv', index=False)

        avg_loss_1 = total_loss_1 / total_samples
        avg_loss_2 = total_loss_2 / total_samples

        return avg_loss_1, avg_loss_2, auroc, f1_score, acc, precision, recall

    elif phase == 'Best Test':
        os.makedirs(f'test_result/fold_{fold + 1}', exist_ok=True)

        result_dict = {'target': y_true, 'pred': y_pred}

        df = pd.DataFrame(result_dict)
        df.to_csv(f'test_result/fold_{fold + 1}/best_test_fold_{fold + 1}.csv', index=False)

        avg_loss_1 = total_loss_1 / total_samples
        avg_loss_2 = total_loss_2 / total_samples

        return avg_loss_1, avg_loss_2, auroc, f1_score, acc, precision, recall
    
    else:
        os.makedirs(f'validation_result/fold_{fold + 1}', exist_ok=True)

        result_dict = {'target': y_true, 'pred': y_pred}

        df = pd.DataFrame(result_dict)
        df.to_csv(f'validation_result/fold_{fold + 1}/validation_epoch_{epoch + 1}.csv', index=False)

        avg_loss_1 = total_loss_1 / total_samples
        avg_loss_2 = total_loss_2 / total_samples
        avg_loss = total_loss / total_samples

        return avg_loss, avg_loss_1, avg_loss_2, auroc, f1_score, acc, precision, recall


def test_model(model, data_loader, criterion_1, criterion_2, fold):
        final_path = f'model_check_point/fold_{fold + 1}/final_model_fold_{fold + 1}.pth'

        # Load the best model weights
        model.load_state_dict(torch.load(final_path))
        final_loss_1, final_loss_2, final_auroc, final_f1_score, final_acc, final_precision, final_recall = evaluate_model(model, data_loader, criterion_1, criterion_2, fold=fold, phase='Final Test')

        # Print the evaluation metrics
        print("Final Instance Loss:", final_loss_1)
        print("Final Bag Loss:", final_loss_2)
        print("Final AUROC:", final_auroc)
        print("Final F1 Score:", final_f1_score)
        print("Final Accuracy:", final_acc)
        print("Final Precision:", final_precision)
        print("Final Recall:", final_recall)

        best_path = f'model_check_point/fold_{fold + 1}/best_model_fold_{fold + 1}.pth'
        # Load the best model weights
        model.load_state_dict(torch.load(best_path))
        best_loss_1, best_loss_2, best_auroc, best_f1_score, best_acc, best_precision, best_recall = evaluate_model(model, data_loader, criterion_1, criterion_2, fold=fold, phase='Best Test')

        print("Best Instance Loss:", best_loss_1)
        print("Best Bag Loss:", best_loss_2)
        print("Best AUROC:", best_auroc)
        print("Best F1 Score:", best_f1_score)
        print("Best Accuracy:", best_acc)
        print("Best Precision:", best_precision)
        print("Best Recall:", best_recall)
