"""Independent bounded conditional delta checks for two sensitivity states.

Author: Amanjeet Singh
The preferred estimator integrates the final one-day stock-measure exercise
probability analytically. No variance is clipped and no observation is dropped.
If a non-finite state is encountered, that estimator is marked unverified and
the complete sample is rerun with exact exercise indicators and a pre-drawn
Gaussian path bound. This script never builds a deterministic delta grid.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr

__author__ = "Amanjeet Singh"


class NumericalExplosion(RuntimeError):
    pass


def summary(total, total2, n, discount):
    mean = total/n
    se = np.sqrt(max((total2/n-mean**2)/(n-1), 0.0))
    return {"estimate":float(mean),"standard_error":float(se),
            "ci95":[float(mean-1.96*se),float(mean+1.96*se)],
            "sample_size":n,"delta_bound":[0.0,float(discount)]}


def rao_blackwell(params, seed, n=500_000, block_size=50_000, days=63,
                  r=.0167, q=.0165, S=100., K=100.):
    rng=np.random.default_rng(seed)
    carry=(r-q)/252
    discount=np.exp(-q*days/252)
    total=total2=0.0
    max_h=0.0
    started=time.monotonic()
    for offset in range(0,n,block_size):
        m=min(block_size,n-offset)
        x=np.full(m,np.log(S/K),dtype=np.float64)
        h=np.full(m,params['initial_variance'],dtype=np.float64)
        with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
            for t in range(days-1):
                u=rng.standard_normal(m)
                root_h=np.sqrt(h)
                x+=carry+h/2+root_h*u
                # u+sqrt(h)-eta(h), expressed without subtracting two large
                # square-root terms. This is exactly the paper's transition.
                eps=u+root_h/2-(params['mu']-carry)/root_h
                h=(params['omega']+h*(params['alpha']+params['gamma']*(eps<0))*eps**2
                   +params['beta']*h)
                if not (np.isfinite(x).all() and np.isfinite(h).all() and (h>0).all()):
                    raise NumericalExplosion(f"Non-finite or non-positive state in block {offset//block_size}, day {t+1}")
                max_h=max(max_h,float(h.max()))
            values=discount*ndtr((x+carry+h/2)/np.sqrt(h))
        if not (np.isfinite(values).all() and (values>=0).all() and (values<=discount).all()):
            raise NumericalExplosion("Final conditional probabilities invalid")
        total+=float(values.sum())
        total2+=float(np.dot(values,values))
    return {**summary(total,total2,n,discount),"verified":True,
            "method":"stock-numeraire conditional Monte Carlo with analytic last-day integration",
            "seed":seed,"rng":"NumPy default_rng (PCG64)",
            "rng_draw_order":"50,000-path blocks; independent normals by date within each block",
            "maximum_block_paths":block_size,"simulated_periods":days-1,
            "last_day_integrated_analytically":True,"paths_dropped":0,
            "variance_clipped":False,"pathwise_classifications":0,
            "maximum_simulated_variance":max_h,"seconds":time.monotonic()-started}


def exact_indicator_with_path_bound(params, seed, n=500_000, block_size=50_000,
                                    days=63, r=.0167, q=.0165,S=100.,K=100.):
    """Exact exercise classification from pre-drawn finite Gaussian vectors.

    For every positive h, c+h/2+sqrt(h)*u >= c-u^2/2. If B=max|u| over the
    pre-drawn vector and m periods remain, x+m*c-m*B^2/2>0 guarantees exercise
    irrespective of every subsequent variance state. Classified observations
    remain in the sample with indicator one; no variance cap is introduced.
    """
    rng=np.random.default_rng(seed)
    carry=(r-q)/252;discount=np.exp(-q*days/252)
    total=total2=0.;early_count=0;max_h=0.;started=time.monotonic()
    for offset in range(0,n,block_size):
        m=min(block_size,n-offset)
        shocks=rng.standard_normal((m,days))
        bound=np.maximum(shocks.max(axis=1),-shocks.min(axis=1))**2
        x=np.full(m,np.log(S/K));h=np.full(m,params['initial_variance'])
        alive=np.ones(m,dtype=bool);exercises=np.zeros(m,dtype=bool)
        for t in range(days):
            ids=np.flatnonzero(alive)
            if not len(ids):break
            hh=h[ids];u=shocks[ids,t];root_h=np.sqrt(hh)
            with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                x[ids]+=carry+hh/2+root_h*u
                remaining=days-t-1
                lower=x[ids]+remaining*carry-remaining*bound[ids]/2
                # Conservative comparison margin; close cases continue through
                # the unmodified variance recursion instead of being classified.
                scale=1+np.abs(x[ids])+remaining*bound[ids]/2
                certain=lower>1e-10*scale
                if remaining==0:certain=x[ids]>0
                exercises[ids[certain]]=True
                alive[ids[certain]]=False
                if remaining:
                    early_count+=int(certain.sum())
                    keep=ids[~certain];hk=h[keep];sk=np.sqrt(hk)
                    eps=shocks[keep,t]+sk/2-(params['mu']-carry)/sk
                    h[keep]=(params['omega']+hk*(params['alpha']+params['gamma']*(eps<0))*eps**2
                             +params['beta']*hk)
                    if not(np.isfinite(h[keep]).all() and (h[keep]>0).all() and np.isfinite(x[keep]).all()):
                        raise NumericalExplosion("Unverified state despite pathwise exercise bound")
                    if len(keep):max_h=max(max_h,float(h[keep].max()))
        values=discount*exercises.astype(float)
        total+=float(values.sum());total2+=float(np.dot(values,values))
    return {**summary(total,total2,n,discount),"verified":True,
            "method":"stock-numeraire exact exercise indicators with pre-drawn Gaussian future bound",
            "seed":seed,"rng":"NumPy default_rng (PCG64)",
            "rng_draw_order":"50,000-path blocks; 63 normals per path pre-drawn in row-major order",
            "maximum_block_paths":block_size,"simulated_periods":days,
            "last_day_integrated_analytically":False,"paths_dropped":0,
            "variance_clipped":False,"pathwise_classifications":early_count,
            "pathwise_bound":"R_j=c+h_j/2+sqrt(h_j)u_j >= c-u_j^2/2; if x+m*c-m*max_j|u_j|^2/2>0, final log moneyness is positive for that realised Gaussian vector.",
            "maximum_simulated_variance":max_h,"seconds":time.monotonic()-started}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=Path('calibration_sensitivity'))
    args=parser.parse_args()
    here=args.folder.resolve()
    cases=[]
    for name,dirname,seed in (
        ('raw_coefficients_price_compatible_initial_variance','price_compatible',70461),
        ('scaled_fit_fitted_initial_variance','scaled_fit',70462)):
        calibration=json.loads((here/dirname/'calibration.json').read_text())
        params=calibration['selected_parameters']
        failure=None
        try:
            result=rao_blackwell(params,seed)
        except (FloatingPointError,NumericalExplosion) as e:
            failure={'verified':False,'attempted_method':'analytic last-day stock-measure simulation',
                     'reason':str(e),'estimate_reported':False}
            result=exact_indicator_with_path_bound(params,seed)
        row={'case':name,'parameters':params,'state':{'spot':100.,'strike':100.,'days':63},
             'rates':{'r':.0167,'q':.0165},**result,'grid_delta':None,
             'grid_minus_mc':None,'difference_in_mc_standard_errors':None}
        saved=here/'sensitivity_results.json'
        if saved.is_file():
            report=json.loads(saved.read_text())
            case=next(c for c in report['cases'] if c['name']==dirname)
            initial=case['policies'][0].get('initial_conditional_delta')
            if initial is not None:
                row['grid_delta']=initial['estimate']
                row['grid_minus_mc']=initial['estimate']-row['estimate']
                row['difference_in_mc_standard_errors']=row['grid_minus_mc']/row['standard_error']
        if failure:row['rao_blackwell_failure']=failure
        cases.append(row)
        print(json.dumps(row,indent=2),flush=True)
        out={'author':__author__,'purpose':'Two independent conditional-delta accuracy diagnostics for calibration sensitivities; these are not global error bounds.',
             'stock_numeraire_identity':'Delta=e^(-q*tau/252) Q^S(ST>K); Z=sqrt(h)+u under Q^S with u standard normal; the shifted GJR variance transition is retained.',
             'cases':cases}
        (here/'delta_sensitivity_checks.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
