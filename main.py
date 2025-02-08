import argparse
import copy
import sys
import time
import random
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

sys.path.insert(0, './src')
from datasets.PPIMI_datasets import ModulatorPPIDataset, performance_evaluation
from cross_attention_ppimi import CrossAttentionPPIMI

def ensure_dir(directory):
    """Ensure directory exists, create if not exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_metrics_to_file(metrics, file_path, mode='a'):
    """Save metrics to file, ensure directory exists"""
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
    
    with open(file_path, mode) as f:
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 50 + "\n")

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """Save model checkpoint"""
    directory = os.path.dirname(filename)
    ensure_dir(directory)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)

def train(PPIMI_model, device, dataloader, optimizer, criterion):
    """Standard training function"""
    PPIMI_model.train()
    total_loss = 0
    ce_loss_total = 0
    l2_loss_total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        fingerprint, protein_feats, cls_repr, atomic_reprs, label = batch
        
        fingerprint = fingerprint.to(device)
        protein_feats = protein_feats.to(device)
        cls_repr = cls_repr.to(device)
        atomic_reprs = atomic_reprs.to(device)
        label = label.to(device)

        pred, l2_loss = PPIMI_model(
            modulator=atomic_reprs,
            fingerprints=fingerprint,
            ppi_feats=protein_feats,
            domain_label=False
        )
        
        pred = pred.view(-1, 2)
        label = label.view(-1)
        
        ce_loss = criterion(pred, label)
        loss = ce_loss + l2_loss
        
        flooded_loss = torch.abs(loss - FLOOD_LEVEL) + FLOOD_LEVEL
        final_loss = flooded_loss

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(PPIMI_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        ce_loss_total += ce_loss.item()
        l2_loss_total += l2_loss.item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce_loss': f'{ce_loss.item():.4f}',
            'l2_loss': f'{l2_loss.item():.4f}'
        })

    return total_loss / len(dataloader)

def train_with_domain_adaptation(PPIMI_model, device, source_dataloader, target_unlabeled_dataloader, 
                               optimizer_feature, optimizer_domain, criterion, domain_criterion, domain_lambda=0.1):
    """Domain adaptation training function"""
    PPIMI_model.train()
    total_loss = 0
    ce_loss_total = 0
    domain_loss_total = 0
    
    target_iter = iter(target_unlabeled_dataloader)
    
    progress_bar = tqdm(source_dataloader, desc="Training with DA")
    for batch_idx, source_batch in enumerate(progress_bar):
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_unlabeled_dataloader)
            target_batch = next(target_iter)
            
        # Source domain data
        s_fingerprint, s_protein_feats, s_cls_repr, s_atomic_reprs, s_label = source_batch
        s_fingerprint = s_fingerprint.to(device)
        s_protein_feats = s_protein_feats.to(device)
        s_cls_repr = s_cls_repr.to(device)
        s_atomic_reprs = s_atomic_reprs.to(device)
        s_label = s_label.to(device)

        # Target domain data
        t_fingerprint, t_protein_feats, t_cls_repr, t_atomic_reprs, _ = target_batch
        t_fingerprint = t_fingerprint.to(device)
        t_protein_feats = t_protein_feats.to(device)
        t_cls_repr = t_cls_repr.to(device)
        t_atomic_reprs = t_atomic_reprs.to(device)
        
        # Domain labels
        s_domain_label = torch.ones(s_fingerprint.size(0), 1).to(device)
        t_domain_label = torch.zeros(t_fingerprint.size(0), 1).to(device)
        
        # 1. Train domain discriminator
        optimizer_domain.zero_grad()
        
        s_logits, s_domain_pred = PPIMI_model(
            modulator=s_atomic_reprs,
            fingerprints=s_fingerprint,
            ppi_feats=s_protein_feats,
            domain_label=True
        )
        
        t_logits, t_domain_pred = PPIMI_model(
            modulator=t_atomic_reprs,
            fingerprints=t_fingerprint,
            ppi_feats=t_protein_feats,
            domain_label=True
        )
        
        domain_pred = torch.cat([s_domain_pred, t_domain_pred], dim=0)
        domain_label = torch.cat([s_domain_label, t_domain_label], dim=0)
        d_loss = domain_criterion(domain_pred, domain_label)
        d_loss.backward()
        optimizer_domain.step()
        
        # 2. Train feature extractor and classifier
        optimizer_feature.zero_grad()
        
        s_logits, s_domain_pred = PPIMI_model(
            modulator=s_atomic_reprs,
            fingerprints=s_fingerprint,
            ppi_feats=s_protein_feats,
            domain_label=True
        )
        t_logits, t_domain_pred = PPIMI_model(
            modulator=t_atomic_reprs,
            fingerprints=t_fingerprint,
            ppi_feats=t_protein_feats,
            domain_label=True
        )
        
        s_logits = s_logits.view(-1, 2)
        s_label = s_label.view(-1)
        ce_loss = criterion(s_logits, s_label)
        
        domain_pred = torch.cat([s_domain_pred, t_domain_pred], dim=0)
        domain_label = torch.cat([s_domain_label, t_domain_label], dim=0)
        domain_loss = domain_criterion(domain_pred, domain_label)
        
        total_main_loss = ce_loss + domain_lambda * domain_loss
        flooded_loss = torch.abs(total_main_loss - FLOOD_LEVEL) + FLOOD_LEVEL
        final_loss = flooded_loss

        final_loss.backward()
        optimizer_feature.step()
        
        total_loss += total_main_loss.item()
        ce_loss_total += ce_loss.item()
        domain_loss_total += domain_loss.item()
        
        progress_bar.set_postfix({
            'total_loss': f'{total_main_loss.item():.4f}',
            'ce_loss': f'{ce_loss.item():.4f}',
            'domain_loss': f'{domain_loss.item():.4f}'
        })
    
    return (total_loss / len(source_dataloader), 
            ce_loss_total / len(source_dataloader), 
            domain_loss_total / len(source_dataloader))

def predicting(PPIMI_model, device, dataloader):
    """Prediction function"""
    PPIMI_model.eval()
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Predicting")
        for batch_idx, batch in enumerate(progress_bar):
            if len(batch) != 5:
                print(f"Unexpected batch size: {len(batch)}")
                raise ValueError(f"Batch size is {len(batch)}, expected 5.")
            
            fingerprint, protein_feats, cls_repr, atomic_reprs, label = batch
            fingerprint = fingerprint.to(device)
            protein_feats = protein_feats.to(device)
            cls_repr = cls_repr.to(device)
            atomic_reprs = atomic_reprs.to(device)
            label = label.to(device)
            
            pred, _ = PPIMI_model(
                modulator=atomic_reprs,
                fingerprints=fingerprint,
                ppi_feats=protein_feats,
                domain_label=False
            )
            pred = pred.squeeze()
            
            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
                
            total_preds.append(pred.detach().cpu())
            total_labels.append(label.detach().cpu())
    
    total_preds = torch.cat(total_preds, dim=0).numpy()
    total_labels = torch.cat(total_labels, dim=0).numpy()
    
    return total_labels, total_preds

def validate_args(args):
    """Validate command line arguments"""
    assert args.batch_size > 0, "Batch size must be positive"
    assert args.learning_rate > 0, "Learning rate must be positive"
    assert args.epochs > 0, "Number of epochs must be positive"
    assert 0 <= args.dropout <= 1, "Dropout ratio must be between 0 and 1"
    assert args.eval_setting in ['random', 'cold'], "Invalid evaluation setting"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of AlphaPPIMI')
    # Basic parameters
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_setting', type=str, default='random', 
                       choices=['random', 'cold'],
                       help='Evaluation setting: random split or cold-start')
    parser.add_argument('--fold', type=str, required=True, 
                       help='Fold number for cross validation')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--runseed', type=int, default=124)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--out_path', type=str, default='./outputs/')

    # Model parameters
    parser.add_argument('--ppi_hidden_dim', type=int, default=3366,
                       help='Hidden dimension for PPI features')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of heads in cross attention')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                       help='Number of cross attention layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Domain adaptation parameters
    parser.add_argument('--use_domain_adaptation', action='store_true',
                       help='Whether to use domain adaptation')
    parser.add_argument('--target_dataset', type=str, default='DiPPI',
                       choices=['DiPPI', 'iPPIDB'],
                       help='Target domain dataset selection')
    parser.add_argument('--domain_lambda', type=float, default=0.1,
                       help='Weight for domain adaptation loss')
    
    args = parser.parse_args()
    print(args)

    try:
        validate_args(args)
        os.makedirs(args.out_path, exist_ok=True)
        print(f"Output directory is set to: {args.out_path}")

        # Set random seeds
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load datasets
        if args.use_domain_adaptation:
            args.out_path = os.path.join('./outputs/domain_adaptation/', args.target_dataset)
            ensure_dir(args.out_path)
            print(f"Output directory set to: {args.out_path}")

            source_data_path = './data/domain_adaptation/source/'
            print(f"Loading datasets for domain adaptation, target domain: {args.target_dataset}")
            train_source_dataset = ModulatorPPIDataset(
                mode='train', setting=args.eval_setting, fold=args.fold,
                domain='source', use_domain_adaptation=True,
                data_path=source_data_path
            )

            val_source_dataset = ModulatorPPIDataset(
                mode='valid', setting=args.eval_setting, fold=args.fold,
                domain='source', use_domain_adaptation=True,
                data_path=source_data_path
            )

            target_data_path = f'./data/domain_adaptation/target/{args.target_dataset}/'
            if not os.path.exists(target_data_path):
                raise FileNotFoundError(f"Target domain directory not found: {target_data_path}")

            train_target_unlabeled_dataset = ModulatorPPIDataset(
                mode='train_unlabeled', setting=None, fold=None,
                domain='target', use_domain_adaptation=True,
                data_path=target_data_path
            )

            test_target_dataset = ModulatorPPIDataset(
                mode='test', setting=None, fold=None,
                domain='target', use_domain_adaptation=True,
                data_path=target_data_path
            )

            # Create DataLoaders
            train_source_dataloader = DataLoader(
                train_source_dataset, batch_size=args.batch_size,
                shuffle=True, drop_last=True
            )
            val_source_dataloader = DataLoader(
                val_source_dataset, batch_size=args.batch_size,
                shuffle=False
            )
            train_target_unlabeled_dataloader = DataLoader(
                train_target_unlabeled_dataset, batch_size=args.batch_size,
                shuffle=True, drop_last=True
            )
            test_target_dataloader = DataLoader(
                test_target_dataset, batch_size=args.batch_size,
                shuffle=False
            )

        else:
            args.out_path = os.path.join('./outputs/standard/', args.eval_setting)
            ensure_dir(args.out_path)
            print(f"Output directory set to: {args.out_path}")

            data_path = './data/standard/'
            print(f"Loading standard datasets with {args.eval_setting} split")
            train_dataset = ModulatorPPIDataset(
                mode='train', setting=args.eval_setting, fold=args.fold,
                use_domain_adaptation=False, data_path=data_path
            )
            val_dataset = ModulatorPPIDataset(
                mode='valid', setting=args.eval_setting, fold=args.fold,
                use_domain_adaptation=False, data_path=data_path
            )
            test_dataset = ModulatorPPIDataset(
                mode='test', setting=args.eval_setting, fold=args.fold,
                use_domain_adaptation=False, data_path=data_path
            )

            # Create DataLoaders
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size,
                shuffle=True, drop_last=True
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=False
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size,
                shuffle=False
            )

        # Initialize model
        PPIMI_model = CrossAttentionPPIMI(
            modulator_emb_dim=768,
            fingerprint_dim=1024,
            ppi_emb_dim=args.ppi_hidden_dim,
            nhead=args.nhead,
            num_cross_layers=args.num_encoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            grl_lambda=1.0
        ).to(device)

        # Loss functions and optimizers
        criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCELoss()
        optimizer_feature = torch.optim.AdamW(
            list(PPIMI_model.classifier.parameters()), 
            lr=args.learning_rate
        )

        if args.use_domain_adaptation:
            optimizer_domain = torch.optim.AdamW(
                PPIMI_model.domain_discriminator.parameters(),
                lr=args.learning_rate * 10
            )
        else:
            optimizer_domain = None

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_feature, mode='max', factor=0.5, patience=10, verbose=True
        )

        # Training variables
        best_model = None
        best_roc_auc = 0
        best_epoch = 0
        early_stopping_patience = 50
        no_improve_count = 0
        FLOOD_LEVEL = 0.13

        train_start_time = time.time()
        
        try:
            for epoch in range(1, args.epochs + 1):
                print(f"Starting epoch {epoch}")
                epoch_start_time = time.time()
                print(f'\n{"="*50}')
                print(f'Epoch {epoch}/{args.epochs}')
                
                if args.use_domain_adaptation:
                    train_loss, ce_loss, domain_loss = train_with_domain_adaptation(
                        PPIMI_model, device, train_source_dataloader, 
                        train_target_unlabeled_dataloader, optimizer_feature, optimizer_domain, 
                        criterion, domain_criterion, domain_lambda=args.domain_lambda
                    )
                    print(f'\nTraining Losses:')
                    print(f'Total Loss: {train_loss:.4f}')
                    print(f'CE Loss: {ce_loss:.4f}')
                    print(f'Domain Loss: {domain_loss:.4f}')
                    
                    G, P = predicting(PPIMI_model, device, test_target_dataloader)
                    eval_set = 'target'
                else:
                    train_loss = train(PPIMI_model, device, train_dataloader, optimizer_feature, criterion)
                    print(f'\nTraining Loss: {train_loss:.4f}')
                    
                    G, P = predicting(PPIMI_model, device, val_dataloader)
                    eval_set = 'source'
                
                # Calculate metrics
                current_roc_auc, current_aupr, precision, accuracy, recall, f1, specificity, mcc, pred_labels = performance_evaluation(P, G)
                
                print(f'ROC-AUC:     {current_roc_auc:.4f}')
                print(f'AUPR:        {current_aupr:.4f}')
                print(f'Precision:   {precision:.4f}')
                print(f'Accuracy:    {accuracy:.4f}')
                print(f'Recall:      {recall:.4f}')
                print(f'F1 Score:    {f1:.4f}')
                print(f'Specificity: {specificity:.4f}')
                print(f'MCC:         {mcc:.4f}')
                
                current_lr = optimizer_feature.param_groups[0]['lr']
                print(f'Learning Rate: {current_lr:.2e}')
                
                # Save validation metrics
                val_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'roc_auc': current_roc_auc,
                    'aupr': current_aupr,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f1': f1,
                    'specificity': specificity,
                    'mcc': mcc
                }
                save_metrics_to_file(val_metrics, f"{args.out_path}/validation_metrics_{eval_set}.txt")
                
                scheduler.step(current_roc_auc)
                
                # Model selection and early stopping
                if current_roc_auc > best_roc_auc:
                    best_model = copy.deepcopy(PPIMI_model)
                    best_roc_auc = current_roc_auc
                    best_epoch = epoch
                    no_improve_count = 0
                    
                    checkpoint_path = f"{args.out_path}/best_checkpoint.pt"
                    save_checkpoint(PPIMI_model, optimizer_feature, epoch, val_metrics, checkpoint_path)
                    print(f'\nNew best model saved! (ROC-AUC: {current_roc_auc:.4f})')
                else:
                    no_improve_count += 1
                    print(f'\nNo improvement for {no_improve_count} epochs (Best ROC-AUC: {best_roc_auc:.4f} at epoch {best_epoch})')
                    if no_improve_count >= early_stopping_patience:
                        print(f'Early stopping triggered after {epoch} epochs')
                        break
                
                epoch_time = time.time() - epoch_start_time
                print(f'Epoch time: {epoch_time:.2f}s')

            # Evaluate final model
            if args.use_domain_adaptation:
                G, P = predicting(PPIMI_model, device, test_target_dataloader)
                final_eval_set = 'target'
            else:
                G, P = predicting(PPIMI_model, device, test_dataloader)
                final_eval_set = 'source'

            final_metrics = performance_evaluation(P, G)[:-1]
            metrics_dict = {
                'roc_auc': final_metrics[0],
                'aupr': final_metrics[1],
                'precision': final_metrics[2],
                'accuracy': final_metrics[3],
                'recall': final_metrics[4],
                'f1': final_metrics[5],
                'specificity': final_metrics[6],
                'mcc': final_metrics[7]
            }

            print('\nFinal Model Test Results:')
            for metric_name, value in metrics_dict.items():
                print(f'{metric_name.upper():12s}: {value:.4f}')
            
            save_metrics_to_file(
                {'final_model': metrics_dict}, 
                f"{args.out_path}/test_metrics_{final_eval_set}.txt"
            )

            # Save best model
            model_path = f"{args.out_path}/setting_{args.eval_setting}_fold{args.fold}.model"
            torch.save(best_model.state_dict(), model_path)
            print(f'\nBest model saved to {model_path}')

        except Exception as e:
            print(f"Error occurred during training: {str(e)}")
            if 'val_metrics' in locals():
                emergency_path = f"{args.out_path}/emergency_checkpoint.pt"
                save_checkpoint(PPIMI_model, optimizer_feature, epoch, val_metrics, emergency_path)
                print(f"Emergency checkpoint saved to {emergency_path}")
            raise e

        finally:
            total_time = (time.time() - train_start_time) / 3600
            print(f'\nTotal training time: {total_time:.2f} hours')

    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)