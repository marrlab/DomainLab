#!/bin/bash

# Phase portraits
python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="x-axis=loss_ell task vs y-axis=loss                 r/dyngamma_d" --plot1="loss_task/ell" --legend2="x-axis=loss_ell task vs y-axis=loss r/dyngamma_d" --legend1="loss_task/ell" --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="x-axis=loss_ell task vs y-axis=loss                 r/dynmu_recon" --plot1="loss_task/ell" --legend2="x-axis=loss_ell task vs y-axis=loss r/dynmu_recon" --legend1="loss_task/ell" --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="x-axis=loss_ell task vs y-axis=loss                 r/dynbeta_d" --plot1="loss_task/ell" --legend2="x-axis=loss_ell task vs y-axis=loss r/dynbeta_d" --legend1="loss_task/ell" --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="x-axis=loss_ell task vs y-axis=loss                 r/dynbeta_x" --plot1="loss_task/ell" --legend2="x-axis=loss_ell task vs y-axis=loss r/dynbeta_x" --legend1="loss_task/ell" --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="x-axis=loss_ell task vs y-axis=loss                 r/dynbeta_y" --plot1="loss_task/ell" --legend2="x-axis=loss_ell task vs y-axis=loss r/dynbeta_y" --legend1="loss_task/ell" --output_dir="./figures_diva" --phase_portrait


# Plot R and the corresponding set point curves (both in the same figure)
python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_gamma_d" --plot2="lossrs/setpoint_gamma_d" --legend1="loss dynamic corresponding to multiplier dyn_gamma_d" --legend2="corresponding setpoint_gamma_d" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_mu_recon" --plot2="lossrs/setpoint_mu_recon" --legend1="loss dynamic corresponding to multiplier dyn_mu_recon" --legend2="corresponding setpoint_mu_recon" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_beta_d" --plot2="lossrs/setpoint_beta_d" --legend1="loss dynamic corresponding to multiplier dyn_beta_d" --legend2="corresponding setpoint_beta_d" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_beta_x" --plot2="lossrs/setpoint_beta_x" --legend1="loss dynamic corresponding to multiplier dyn_beta_x" --legend2="corresponding setpoint_beta_x" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_beta_y" --plot2="lossrs/setpoint_beta_y" --legend1="loss dynamic corresponding to multiplier dyn_beta_y" --legend2="corresponding setpoint_beta_y" --output_dir="./figures_diva"


# Other plots (one curve per figure)
values=('controller_gain/beta_d' 'controller_gain/beta_y' 'controller_gain/beta_x' 'controller_gain/gamma_d' 'controller_gain/mu_recon' 'dyn_mu/beta_d' 'delta/beta_d' 'dyn_mu/beta_y' 'delta/beta_y' 'dyn_mu/beta_x' 'delta/beta_x' 'dyn_mu/gamma_d' 'delta/gamma_d' 'dyn_mu/mu_recon' 'delta/mu_recon' 'loss_task/penalized' 'loss_task/ell' 'acc/te' 'acc/val' 'acc/sel' 'acc/setpoint')
# Loop over the array
for val in "${values[@]}"
do
  python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="$val" --legend1="$val" --output_dir="./figures_diva"
done
