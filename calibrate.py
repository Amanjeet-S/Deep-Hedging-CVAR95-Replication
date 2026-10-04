"""Refit the authors' historical sample, keeping estimation and simulation h0 distinct."""
from __future__ import annotations

__author__ = "Amanjeet Singh"
import argparse
import json
import platform
import warnings
from pathlib import Path
import arch
from arch import arch_model
import numpy as np
import pandas as pd
import scipy


def calibrate(data_path, output_path=None):
    data_path=Path(data_path)
    data=pd.read_csv(data_path,index_col=0).iloc[5031:,:]
    logret=np.log(data.Close/data.Close.shift(1)).dropna()
    if len(logret)!=1259:
        raise ValueError('Expected the published-repository sample of 1,259 log returns')

    def fit(scale):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=arch_model(logret*scale,vol='Garch',p=1,o=1,q=1,rescale=False).fit(disp='off')
        raw={k:float(v) for k,v in result.params.items()}
        decimal=dict(raw);decimal['mu']/=scale;decimal['omega']/=scale**2
        h0=float(result.conditional_volatility.iloc[-1]**2/scale**2)
        residual=float(result.resid.iloc[-1]/scale)
        persistence=decimal['alpha[1]']+decimal['beta[1]']+decimal['gamma[1]']/2
        forecast=decimal['omega']+(decimal['alpha[1]']+decimal['gamma[1]']*(residual<0))*residual**2+decimal['beta[1]']*h0
        return dict(return_scale_before_fit=scale,
            model='Constant-mean Gaussian GJR-GARCH(1,1), arch_model(...p=1,o=1,q=1,rescale=False)',
            params_decimal_returns=decimal,params_estimation_units=raw,
            initial_fitted_variance_h0=h0,initial_fitted_sigma_daily=float(np.sqrt(h0)),
            initial_fitted_sigma_annual=float(np.sqrt(252*h0)),last_fitted_residual=residual,
            next_variance_forecast_from_observed_last_residual=forecast,persistence=persistence,
            stationary_variance_if_exists=float(decimal['omega']/(1-persistence)) if persistence<1 else None,
            loglikelihood_in_estimation_units=float(result.loglikelihood),
            loglikelihood_in_decimal_return_units=float(result.loglikelihood+len(logret)*np.log(scale)),
            convergence_flag=int(result.convergence_flag),optimization_success=bool(result.optimization_result.success),
            optimization_message=str(result.optimization_result.message),
            optimization_iterations=int(result.optimization_result.nit),warnings=[str(w.message) for w in caught])

    out=dict(dataset=str(data_path),date_slice_first_row_before_return=data.index[0],
        first_return_date=logret.index[0],last_return_date=logret.index[-1],n_returns=len(logret),
        sample_mean_daily=float(logret.mean()),sample_sd_daily_ddof1=float(logret.std()),
        sample_sd_annual_ddof1=float(logret.std()*np.sqrt(252)),last_log_return=float(logret.iloc[-1]),
        authors_unscaled_fit=fit(1),scaled_conditioning_control=fit(100),
        library_versions=dict(arch=arch.__version__,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__),
        python_version=platform.python_version())
    fit0=out['authors_unscaled_fit'];p=fit0['params_decimal_returns']
    out['selected_parameters']=dict(mu=p['mu'],omega=p['omega'],alpha=p['alpha[1]'],
        gamma=p['gamma[1]'],beta=p['beta[1]'],initial_variance=fit0['initial_fitted_variance_h0'])
    out['selection_note']='Authors raw-return calibration, with fitted terminal variance used as known first-return variance; paper equations used without artificial initial innovation. Initial hedging capital fixed at published 3.16 for both strategies.'
    if output_path is not None:
        Path(output_path).write_text(json.dumps(out,indent=2)+'\n')
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/SP500_random.csv')
    ap.add_argument('--output',default='recalibration.json');args=ap.parse_args()
    result=calibrate(args.data,args.output)
    print(json.dumps(result['selected_parameters'],indent=2))
