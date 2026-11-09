"""One locked fit. Train/validation only until the best epoch is selected."""
import argparse
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from common import (PACKAGE, RESULTS, atomic_json, atomic_npz, event, require, read, sha, utc,
                    manifest, model_config, verify_package, load_models, load_cache,
                    verify_run, metric_values, next_lr, epoch_order)


def atomic_torch(path, value, torch):
    temp = path.with_name(path.name+'.tmp.%d'%os.getpid())
    with temp.open('wb') as stream:
        torch.save(value,stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temp),str(path))


def run(row, gpu, package=PACKAGE, results=RESULTS):
    require(sys.platform=='linux','Training belongs on configured Linux server, not the local computer')
    lock=verify_package(package,full=False)
    lock_hash=sha(package/'execution_lock.json')
    out=results/'runs'/row['run_id']
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'worker.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').is_file():
            verify_run(out,row,lock_hash)
            print('VERIFIED_EXISTING_PASS '+row['run_id'],flush=True)
            return
        attempt=utc().replace(':','').replace('.','')
        try:
            execute(row,gpu,package,out,lock,lock_hash,attempt)
        except Exception as error:
            failure=dict(status='FAIL',run_id=row['run_id'],attempt=attempt,utc=utc(),
                         error=repr(error),traceback=traceback.format_exc(),execution_lock_sha256=lock_hash)
            atomic_json(out/('failure_'+attempt+'.json'),failure)
            atomic_json(out/'status.json',failure)
            raise


def execute(row,gpu,package,out,lock,lock_hash,attempt):
    # This process only sees its assigned physical GPU, set by the supervisor.
    import torch
    require(torch.cuda.is_available(),'CUDA unavailable')
    require(torch.__version__.split('+')[0]=='2.4.0','Unexpected torch version: '+torch.__version__)
    require(sys.version_info[:2]==(3,8),'Unexpected server Python version')
    device=torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed=int(row['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    modules, imported=load_models(lock['expected_model_sources'])
    model=modules[row['model']].Model(model_config(row)).float().to(device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=float(row['learning_rate']),betas=(0.9,0.999),
                               eps=1e-8,weight_decay=0.,amsgrad=False)
    amp=torch.cuda.amp.GradScaler(enabled=True)
    train=load_cache(package,row,'train')
    validation=load_cache(package,row,'validation')
    batch_size=int(row['batch_size'])

    def tensors(data, index):
        arrays=[data[k][index].copy() for k in ('x_enc','x_mark_enc','x_dec','x_mark_dec')]
        if row['pathway']=='history':
            arrays[2].fill(0)
        x=[torch.as_tensor(a,dtype=torch.float32,device=device) for a in arrays]
        y=torch.as_tensor(data['target_scaled'][index],dtype=torch.float32,device=device)
        return x,y

    def predict(data):
        model.eval(); predictions=[]
        with torch.no_grad():
            for begin in range(0,len(data['origin_ns']),batch_size):
                x,_=tensors(data,slice(begin,begin+batch_size))
                with torch.cuda.amp.autocast(enabled=True):
                    full=model(*x)
                    require(tuple(full.shape)==(len(x[0]),int(row['horizon_hours']),8),'Unexpected output shape')
                    output=full[:,:,-1]
                require(torch.isfinite(output).all().item(),'Nonfinite evaluation output')
                predictions.append(output.float().cpu().numpy())
        return np.concatenate(predictions,axis=0).astype(np.float64)

    state_path=out/'epoch_state.pt'
    epochs=[]; best_value=float('inf'); best_state=None; best_epoch=0; stale=0
    if state_path.is_file():
        checkpoint=torch.load(state_path,map_location='cpu',weights_only=False)
        require(checkpoint['execution_lock_sha256']==lock_hash and checkpoint['run_id']==row['run_id'],'Resume identity mismatch')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp']); epochs=checkpoint['epochs']
        best_state=checkpoint['best_state']; best_value=checkpoint['best_value']
        best_epoch=checkpoint['best_epoch']; stale=checkpoint['stale']
        random.setstate(checkpoint['python_rng']); np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        event(out/'events.jsonl',event='resumed',attempt=attempt,completed_epoch=len(epochs))
        del checkpoint
    config=dict(manifest=row,execution_lock_sha256=lock_hash,physical_gpu=gpu,
                release_manifest_sha256=sha(Path(__file__).resolve().parents[2]/'RELEASE_MANIFEST.json'),
                implementation_scope='release_relocated_runner_not_original_deployment',
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in parameters),
                parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},
                torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
                gpu_name=torch.cuda.get_device_name(0),imported_sources=imported,
                train_windows=len(train['origin_ns']),validation_windows=len(validation['origin_ns']))
    if not (out/'config.json').exists():
        atomic_json(out/'config.json',config)
    event(out/'events.jsonl',event='attempt_start',attempt=attempt,gpu=gpu,
          test_labels_loaded=False,completed_epochs=len(epochs))
    for epoch in range(len(epochs)+1,int(row['max_epochs'])+1):
        if stale>=int(row['patience']):
            break
        start=time.monotonic(); model.train()
        order=epoch_order(len(train['origin_ns']),seed,epoch)
        loss_sum=0.; count=0; skipped=0; updated=0; nonzero_names=set()
        lr=optimizer.param_groups[0]['lr']
        for begin in range(0,len(order),batch_size):
            index=order[begin:begin+batch_size]
            x,y=tensors(train,index)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                full=model(*x)
                require(tuple(full.shape)==(len(index),int(row['horizon_hours']),8),'Unexpected training output shape')
                output=full[:,:,-1]
                loss=torch.nn.functional.mse_loss(output,y,reduction='mean')
            require(torch.isfinite(loss).item(),'Nonfinite training loss; no result-dependent repair')
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            for name,p in model.named_parameters():
                if name not in nonzero_names and p.grad is not None and torch.isfinite(p.grad).all().item() and torch.count_nonzero(p.grad).item()>0:
                    nonzero_names.add(name)
            old_scale=amp.get_scale(); amp.step(optimizer); amp.update()
            if amp.get_scale()<old_scale:
                skipped+=1
            else:
                updated+=1
            loss_sum+=float(loss.detach())*y.numel(); count+=y.numel()
        require(updated>0,'No successful parameter update in epoch')
        require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite model parameters')
        val_pred=predict(validation)
        val_mse=float(np.mean((val_pred-validation['target_scaled'])**2))
        require(np.isfinite(val_mse),'Nonfinite validation MSE')
        better=val_mse<best_value
        if better:
            best_value=val_mse; best_epoch=epoch; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        epochs.append(dict(epoch=epoch,train_mse_std=loss_sum/count,validation_mse_std=val_mse,
                           learning_rate=lr,improved=better,seconds=time.monotonic()-start,
                           optimizer_steps=updated,amp_skips=skipped,nonzero_gradient_parameters=sorted(nonzero_names)))
        for group in optimizer.param_groups:
            group['lr']=next_lr(float(row['learning_rate']),epoch)
        checkpoint=dict(run_id=row['run_id'],execution_lock_sha256=lock_hash,epochs=epochs,
                        model=model.state_dict(),optimizer=optimizer.state_dict(),amp=amp.state_dict(),
                        best_state=best_state,best_value=best_value,best_epoch=best_epoch,stale=stale,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all())
        atomic_torch(state_path,checkpoint,torch)
        atomic_json(out/'epochs.json',epochs)
        status=dict(status='TRAINING',run_id=row['run_id'],epoch=epoch,max_epochs=int(row['max_epochs']),
                    best_epoch=best_epoch,utc=utc(),attempt=attempt,test_labels_loaded=False)
        atomic_json(out/'status.json',status)
        event(out/'events.jsonl',event='epoch_complete',epoch=epoch,seconds=epochs[-1]['seconds'],
              improved=better,amp_skips=skipped,optimizer_steps=updated)
    require(best_state is not None and best_epoch>0,'No selected checkpoint')
    # Always rebuild canonical epoch record from checkpoint state on resume.
    atomic_json(out/'epochs.json',epochs)
    model.load_state_dict(best_state)
    atomic_torch(out/'best_model.pt',dict(model=best_state,epoch=best_epoch,validation_mse_std=best_value,
                                        execution_lock_sha256=lock_hash,run_id=row['run_id']),torch)
    atomic_json(out/'selection.json',dict(best_epoch=best_epoch,validation_mse_std=best_value,
                                        epochs=len(epochs),test_labels_loaded=False,utc=utc()))
    event(out/'events.jsonl',event='selection_complete_before_test',best_epoch=best_epoch)
    test=load_cache(package,row,'test')
    predictions=predict(test)
    scaler=read(package/'train_only_scalers.json')[row['site']]['power']
    scale,mean=scaler['scale'][0],scaler['mean'][0]
    metrics=metric_values(predictions,test['target_scaled'],scale)
    atomic_npz(out/'test_predictions.npz',pred_std=predictions,pred_kw=predictions*scale+mean,
               true_std=test['target_scaled'],true_kw=test['target_kw'],raw_kw=test['target_raw_kw'],
               origin_ns=test['origin_ns'],target_ns=test['target_ns'])
    atomic_json(out/'metrics.json',dict(metrics,test_windows=len(predictions),target_cells=predictions.size,
                                      prediction_postprocessing='none',scale='per_site_train_standardized'))
    verify_package(package,full=False)
    files=['config.json','epochs.json','selection.json','best_model.pt','epoch_state.pt','test_predictions.npz','metrics.json']
    summary=dict(status='PASS',run_id=row['run_id'],execution_lock_sha256=lock_hash,finished_utc=utc(),
                 best_epoch=best_epoch,completed_epochs=len(epochs),test_windows=len(predictions),target_cells=predictions.size,
                 artifacts_sha256={name:sha(out/name) for name in files})
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'status.json',dict(status='PASS',run_id=row['run_id'],utc=utc()))
    event(out/'events.jsonl',event='run_complete',run_id=row['run_id'],best_epoch=best_epoch)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--physical-gpu',required=True)
    parser.add_argument('--package', type=Path, required=True,
                        help='Authorized archived input package; caches are not distributed.')
    parser.add_argument('--results', type=Path, required=True,
                        help='Isolated results directory; do not use original research outputs.')
    parser.add_argument('--allow-training', action='store_true')
    args=parser.parse_args()
    if not args.allow_training:
        parser.error('Training requires explicit --allow-training acknowledgement.')
    if args.results.resolve() == args.package.resolve() or args.results.resolve() in args.package.resolve().parents:
        parser.error('Results cannot be the package directory or its ancestor.')
    rows=[r for r in manifest(args.package) if r['run_id']==args.run_id]
    require(len(rows)==1,'Unknown run ID')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==args.physical_gpu,'GPU visibility must be explicit')
    run(rows[0],args.physical_gpu,args.package,args.results)
