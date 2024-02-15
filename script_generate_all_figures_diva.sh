#!/bin/bash

# Phase portraits
# a command line argument can be passed to this script, in order to skip the first few large jumps on the phase plots; if no argument is provided then all points will be plotted:
# Check if an argument is provided
if [ $# -eq 0 ]; then
  # Check if an argument is provided
  skip_n=0
else
  # Use the provided argument
  skip_n=$1
fi

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="lossrd/dyn_gamma_d" --plot1="loss_task/ell" --legend2="loss (gamma_d)" --legend1="ell" --plot_len 30 --skip_first_n $skip_n --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="lossrd/dyn_mu_recon" --plot1="loss_task/ell" --legend2="reconstruction loss" --legend1="ell" --plot_len 30 --skip_first_n $skip_n --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="lossrd/dyn_beta_d" --plot1="loss_task/ell" --legend2="KL (beta_d)" --legend1="ell" --plot_len 30 --skip_first_n $skip_n --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="lossrd/dyn_beta_x" --plot1="loss_task/ell" --legend2="KL (beta_x)" --legend1="ell" --plot_len 30 --skip_first_n $skip_n --output_dir="./figures_diva" --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --plot2="lossrd/dyn_beta_y" --plot1="loss_task/ell" --legend2="KL (beta_y)" --legend1="ell" --plot_len 30 --skip_first_n $skip_n --output_dir="./figures_diva" --phase_portrait


# Plot R and the corresponding set point curves (both in the same figure)
python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_gamma_d" --plot2="lossrs/setpoint_gamma_d" --legend1="loss (gamma_d)" --legend2="setpoint" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_mu_recon" --plot2="lossrs/setpoint_mu_recon" --legend1="reconstruction loss" --legend2="setpoint" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_beta_d" --plot2="lossrs/setpoint_beta_d" --legend1="KL (beta_d)" --legend2="setpoint" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_beta_x" --plot2="lossrs/setpoint_beta_x" --legend1="KL (beta_x)" --legend2="setpoint" --output_dir="./figures_diva"

python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="lossrd/dyn_beta_y" --plot2="lossrs/setpoint_beta_y" --legend1="KL (beta_y)" --legend2="setpoint" --output_dir="./figures_diva"


 # Other plots (one curve per figure)
 values=('controller_gain/beta_d' 'controller_gain/beta_y' 'controller_gain/beta_x' 'controller_gain/gamma_d' 'controller_gain/mu_recon' 'dyn_mu/beta_d' 'delta/beta_d' 'dyn_mu/beta_y' 'delta/beta_y' 'dyn_mu/beta_x' 'delta/beta_x' 'dyn_mu/gamma_d' 'delta/gamma_d' 'dyn_mu/mu_recon' 'delta/mu_recon' 'loss_task/penalized' 'loss_task/ell' 'acc/te' 'acc/val' 'acc/sel' 'acc/setpoint')
 # Loop over the array
 for val in "${values[@]}"
 do
   python domainlab/utils/generate_fbopt_phase_portrait.py --plot1="$val" --legend1="$val" --output_dir="./figures_diva"
 done
