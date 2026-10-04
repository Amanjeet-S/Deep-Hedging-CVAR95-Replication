"""Independent numerical checks for the 95% experiment implementation."""
__author__ = "Amanjeet Singh"

import json, math
from pathlib import Path
import numpy as np
import torch
from replication import Config, Hedger, physical_paths, cvar, configure_torch

HERE=Path(__file__).resolve().parent
params=json.loads((HERE/'calibration.json').read_text())['selected_parameters']
configure_torch(1)
cfg=Config(maturity=7, training_paths=1000, validation_paths=1000, test_paths=1000)
results={}

def check(name,condition,**details):
    if not condition: raise AssertionError((name,details))
    results[name]={'passed':True,**details}

f,g,p=physical_paths(1000,params,cfg,123)
f2,g2,p2=physical_paths(1000,params,cfg,123)
f3,g3,p3=physical_paths(1000,params,cfg,124)
check('seed_reproducibility',np.array_equal(f,f2) and np.array_equal(g,g2) and np.array_equal(p,p2) and not np.array_equal(g,g3))
check('known_initial_state',np.all(f[:,0,0]==np.float32(np.log(100))) and np.all(f[:,0,2]==np.float32(np.sqrt(252*params['initial_variance']))))

# Independently follow the cash and stock accounts rather than model gains.
s=np.asarray([100.,104.,101.,99.,103.])
small=Config(maturity=4)
fs=np.zeros((1,4,3),np.float32)
fs[0,:,0]=np.log(s[:-1]);fs[0,:,1]=np.arange(4,0,-1)/252;fs[0,:,2]=.1
er,eq=math.exp(small.r/252),math.exp(small.q/252)
gs=(eq*s[1:]-er*s[:-1]).astype(np.float32)[None,:]
for raw in [.3,-.5,1000.]:
    model=Hedger(small)
    for layer in model.policy:
        if isinstance(layer,torch.nn.Linear):
            with torch.no_grad():layer.weight.zero_();layer.bias.zero_()
    with torch.no_grad():model.policy[-1].bias.fill_(raw)
    scripted=torch.jit.script(model)
    with torch.no_grad():
        v,ds,min_cash=scripted.positions(torch.from_numpy(fs),torch.from_numpy(gs),small.initial_capital)
        vf=scripted(torch.from_numpy(fs),torch.from_numpy(gs),small.initial_capital)
    d=ds.numpy()[0].astype(float)
    account=small.initial_capital
    cash_min=math.inf
    for t in range(4):
        cash=account-d[t]*s[t]
        cash_min=min(cash_min,cash)
        account=er*cash+eq*d[t]*s[t+1]
    discounted=er**4*small.initial_capital+sum(d[t]*float(gs[0,t])*er**(3-t) for t in range(4))
    check('cash_account_'+str(raw),abs(account-v.item())<.001 and abs(discounted-v.item())<.001 and torch.equal(v,vf),max_abs_error=max(abs(account-v.item()),abs(discounted-v.item())))
    check('borrowing_constraint_'+str(raw),min_cash.item()>=-100.001 and cash_min>=-100.001,minimum_cash=min_cash.item())

# Derivative of a constant strategy: objective = payoff minus capital and
# terminal stock-gain sum. Compare its autograd derivative on fixed tail paths.
model=Hedger(cfg)
for layer in model.policy:
    if isinstance(layer,torch.nn.Linear):
        with torch.no_grad():layer.weight.zero_();layer.bias.zero_()
with torch.no_grad():model.policy[-1].bias.fill_(.3)
ft,gt,pt=torch.from_numpy(f),torch.from_numpy(g),torch.from_numpy(p)
loss=pt-model(ft,gt,cfg.initial_capital)
obj=torch.topk(loss,50)
obj.values.mean().backward()
autograd=float(model.policy[-1].bias.grad)
terminal_gain=(g.astype(float)*er**np.arange(6,-1,-1)).sum(axis=1)
analytic=-terminal_gain[obj.indices.detach().numpy()].mean()
check('cvar_gradient',abs(autograd-analytic)<1e-5,autograd=autograd,analytic=analytic)
check('cvar_tail',abs(cvar(loss.detach().numpy())-float(obj.values.mean()))<1e-5)

# Wealth cancellation required for the zero-capital overlay, using actual
# trained-policy positions and a different predictable comparison strategy.
torch.manual_seed(1)
model=torch.jit.script(Hedger(cfg))
with torch.no_grad():v,d,_=model.positions(ft,gt,cfg.initial_capital)
d=d.numpy().astype(float)
comparison=np.tile(np.linspace(.4,.7,7),(1000,1))
powers=er**np.arange(6,-1,-1)
v_dh=er**7*cfg.initial_capital+(d*g*powers).sum(axis=1)
v_delta=er**7*cfg.initial_capital+(comparison*g*powers).sum(axis=1)
overlay=((d-comparison)*g*powers).sum(axis=1)
err_dh=p-v_dh;err_delta=p-v_delta
res=np.max(np.abs(overlay-(err_delta-err_dh)))
check('overlay_wealth_identity',res<1e-10,max_absolute_residual=float(res))
check('torchscript_terminal_wealth',np.max(np.abs(v.numpy()-v_dh))<.005,max_absolute_residual=float(np.max(np.abs(v.numpy()-v_dh))))

# Changing future prices/features must not affect positions already chosen.
changed_f=f.copy();changed_g=g.copy();changed_f[:,4:,0]+=.1;changed_g[:,4:]+=3
with torch.no_grad():_,d_changed,_=model.positions(torch.from_numpy(changed_f),torch.from_numpy(changed_g),cfg.initial_capital)
check('predictability',np.array_equal(d[:,:4].astype(np.float32),d_changed.numpy()[:,:4]))
(HERE/'verification.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
