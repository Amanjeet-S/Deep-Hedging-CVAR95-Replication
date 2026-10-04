"""Evaluate a trained 95% agent once on untouched physical-measure paths."""
from __future__ import annotations

__author__ = "Amanjeet Singh"
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from replication import Config,Hedger,physical_paths,cvar,cvar_influence,configure_torch
from delta_reference import DeltaGrid,GJRParameters,GridConfig


def evaluate(folder,grid_path=None,make_plots=True):
    folder=Path(folder)
    ck=torch.load(folder/'agent95.pt',map_location='cpu',weights_only=False)
    cfg=Config(**ck['config']);params=ck['parameters']
    configure_torch(1)
    model=torch.jit.script(Hedger(cfg));model.load_state_dict(ck['state_dict']);model.eval()
    print('Generating final independent test sample',flush=True)
    f,g,pay=physical_paths(cfg.test_paths,params,cfg,cfg.test_seed)
    torch_wealth=[]; dh=[]; min_cash=[]
    with torch.no_grad():
        for j in range(0,cfg.test_paths,cfg.batch_size):
            v,d,mc=model.positions(torch.from_numpy(f[j:j+cfg.batch_size]),torch.from_numpy(g[j:j+cfg.batch_size]),cfg.initial_capital)
            torch_wealth.append(v.numpy());dh.append(d.numpy());min_cash.append(mc.numpy())
    torch_wealth=np.concatenate(torch_wealth)
    dh=np.concatenate(dh).astype(np.float64)
    min_cash=np.concatenate(min_cash)
    if grid_path:
        grid=DeltaGrid.load(grid_path)
        for key in ['mu','omega','alpha','beta','gamma']:
            assert np.isclose(getattr(grid.params,key),params[key],rtol=1e-10),key
        grid.enable_tail_refinement(threshold=.01)
    else:
        p=GJRParameters(**{k:params[k] for k in ['mu','omega','alpha','beta','gamma']},r=cfg.r,q=cfg.q)
        gc=GridConfig(x_min=-1.5,x_max=1.5,nx=4801,logh_min=-12,logh_max=-1,nh=221,quadrature_nodes=48)
        grid=DeltaGrid.build(p,cfg.maturity,gc,progress=lambda t,s:print(f'Delta grid {t}/63 {s:.1f}s',flush=True))
        grid.enable_tail_refinement(threshold=.01)
    n=cfg.test_paths;growth=np.exp(cfg.r/252)
    vdh=np.full(n,cfg.initial_capital);vdelta=vdh.copy();overlay=np.zeros(n)
    current_min_cash=np.full(n,np.inf)
    for t in range(cfg.maturity):
        s=np.exp(f[:,t,0].astype(np.float64))
        h=f[:,t,2].astype(np.float64)**2/252
        delta=grid.delta(cfg.maturity-t,s,h,cfg.strike)
        current_min_cash=np.minimum(current_min_cash,vdh-dh[:,t]*s)
        gg=g[:,t].astype(np.float64)
        vdh=growth*vdh+dh[:,t]*gg
        vdelta=growth*vdelta+delta*gg
        overlay=growth*overlay+(dh[:,t]-delta)*gg
    loss_dh=pay.astype(np.float64)-vdh
    loss_delta=pay.astype(np.float64)-vdelta
    overlay_loss=-overlay
    se=lambda x:float(np.std(x,ddof=1)/np.sqrt(n))
    ifdh=cvar_influence(loss_dh);ifdelta=cvar_influence(loss_delta)
    estimates=[cvar(loss_dh),cvar(loss_dh)-cvar(loss_delta),cvar(overlay_loss),float(overlay.mean())]
    errors=[se(ifdh),se(ifdh-ifdelta),se(cvar_influence(overlay_loss)),se(overlay)]
    published=[3.481,-.081,1.810,-.254]
    names=['Deep hedging CVaR95 loss','Deep minus delta CVaR95 loss','Difference strategy CVaR95 loss','Difference strategy mean P&L']
    metrics=[]
    for name,value,error,target in zip(names,estimates,errors,published):
        metrics.append({'name':name,'published':target,'estimate':value,'standard_error':error,
                        'ci_low':value-1.96*error,'ci_high':value+1.96*error,'difference':value-target})
    checks={'overlay_identity_max_error':float(np.max(np.abs(overlay-(vdh-vdelta)))),
            'hedging_loss_identity_max_error':float(np.max(np.abs(loss_dh-(loss_delta-overlay)))),
            'torch_vs_float64_wealth_max_error':float(np.max(np.abs(vdh-torch_wealth))),
            'minimum_cash_float64':float(current_min_cash.min()),
            'minimum_cash_torch':float(min_cash.min()),
            'borrowing_limit_tolerance':.001,
            'finite_test_losses':bool(np.isfinite(loss_dh).all() and np.isfinite(loss_delta).all()),
            'training_test_distinct_seeds':cfg.training_seed!=cfg.test_seed,
            'validation_test_distinct_seeds':cfg.validation_seed!=cfg.test_seed}
    assert checks['overlay_identity_max_error']<1e-8
    assert checks['hedging_loss_identity_max_error']<1e-8
    assert checks['minimum_cash_float64']>=-cfg.borrowing_limit-.001
    assert checks['finite_test_losses']
    quantiles={str(q):float(np.quantile(overlay,q)) for q in [.001,.01,.05,.5,.95,.99,.999]}
    bins=np.linspace(-5,5,101);counts,_=np.histogram(overlay,bins=bins)
    np.savez_compressed(folder/'terminal_results.npz',deep_loss=loss_dh,delta_loss=loss_delta,overlay_pnl=overlay)
    results={'metrics':metrics,'config':asdict(cfg),'parameters':params,'best_epoch':ck['epoch'],
             'best_validation_cvar':ck['validation_cvar'],'delta_cvar':cvar(loss_delta),
             'delta_standard_error':se(ifdelta),'checks':checks,'delta_grid':grid.diagnostic_summary(),
             'overlay_quantiles':quantiles,'overlay_range':[float(overlay.min()),float(overlay.max())],
             'outside_central_plot':int(np.count_nonzero((overlay<-5)|(overlay>5))),
             'conditional_uncertainty_note':'Sampling intervals condition on one selected policy, fitted/reconstructed parameters and the numerical delta benchmark. They exclude training-seed, parameter and grid uncertainty.',
             'statistical_arbitrage_criterion_met':estimates[2]<0}
    (folder/'results.json').write_text(json.dumps(results,indent=2))
    np.savetxt(folder/'figure1_histogram.csv',np.column_stack((bins[:-1],bins[1:],counts)),delimiter=',',header='bin_left,bin_right,count',comments='')
    print(json.dumps(metrics,indent=2),flush=True)
    if make_plots:plot_results(folder)
    return results


def plot_results(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder)
    results=json.loads((folder/'results.json').read_text())
    with np.load(folder/'terminal_results.npz') as a:pnl=a['overlay_pnl']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),gridspec_kw={'width_ratios':[1.3,1]})
    color='#279e68'
    axes[0].hist(pnl,bins=np.linspace(-5,5,101),color=color,alpha=.82,label='CVaR 95% agent')
    axes[0].set_xlim(-5,5);axes[0].set_title('Central range: -5 to 5')
    axes[1].hist(pnl,bins=100,color=color,alpha=.82)
    axes[1].set_title('Full simulated range');axes[1].set_yscale('log')
    for ax in axes:
        ax.axvline(0,color='#333333',lw=.8)
        ax.axvline(pnl.mean(),color='#17384f',ls='--',lw=1.2)
        ax.set_xlabel('Difference-strategy terminal P&L');ax.set_ylabel('Frequency (paths)')
        ax.grid(axis='y',alpha=.16)
    axes[0].legend(frameon=False)
    fig.suptitle('Independent distribution - 95% agent',fontsize=14,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(folder/'figure1_95.png',dpi=190, metadata={'Author': __author__});plt.close(fig)
    h=json.loads((folder/'training_history.json').read_text())
    fig,ax=plt.subplots(figsize=(10,3.0))
    epochs=[v['epoch'] for v in h]
    ax.plot(epochs,[v['training_cvar'] for v in h],label='Training (changing policy within epoch)',alpha=.65,color='#7b8da2')
    ax.plot(epochs,[v['validation_cvar'] for v in h],label='Separate validation',color='#256c88')
    ax.scatter([results['best_epoch']],[results['best_validation_cvar']],color='#279e68',zorder=3,label='Selected checkpoint')
    ax.set_xlabel('Epoch');ax.set_ylabel('CVaR 95% hedging loss');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(folder/'training_curve.png',dpi=170, metadata={'Author': __author__});plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);ap.add_argument('--grid')
    a=ap.parse_args();evaluate(a.folder,a.grid)
