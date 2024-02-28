#!/bin/bash -x -v

STR_LOSS_ELL="loss_task/ell"
OUT_DIR="./figures_diva"
# Number of points to plot:
phase_portrait_plot_len=120

LOSS_GAMMA_D="$\mathbb{E}_{q_{\phi_d}(z_d|x)}[\log q_{\omega_d}(d|z_d)]$"


# README:
# The following scripts will check event files from the 'runs' folder of the working directory.
# To generate example tensorboard 'runs' folder, one could execute e.g. `sh run_fbopt_mnist_diva_autoki.sh` such that there will be 'runs' folder.

if [ -z "$1" ]; then
  # Check if an argument is provided
  runs_dir="runs/*"
else
  # Use the provided argument
  runs_dir=$1
fi


# a command line argument can be passed to this script, in order to skip the first few large jumps on the phase plots; if no argument is provided then all points will be plotted:
if [ -z "$2" ]; then
  # Check if an argument is provided
  skip_n=0
else
  # Use the provided argument
  skip_n=$2
fi




# Phase portraits
python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot2="lossrd/dyn_gamma_d" --plot1="loss_task/ell" --legend2="\$R_{\gamma_d}(\cdot)\$" --legend1="\$\ell(\cdot)\$" --plot_len $phase_portrait_plot_len --skip_n_steps $skip_n --output_dir=$OUT_DIR --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot2="lossrd/dyn_mu_recon" --plot1="loss_task/ell" --legend2="reconstruction loss" --legend1="\$\ell(\cdot)\$" --plot_len $phase_portrait_plot_len --skip_n_steps $skip_n --output_dir=$OUT_DIR --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot2="lossrd/dyn_beta_d" --plot1="loss_task/ell" --legend2="KL (beta_d)" --legend1="ell" --plot_len $phase_portrait_plot_len --skip_n_steps $skip_n --output_dir=$OUT_DIR --phase_portrait

# python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot2="lossrd/dyn_beta_x" --plot1="loss_task/ell" --legend2="KL (beta_x)" --legend1="ell" --plot_len $phase_portrait_plot_len --skip_n_steps $skip_n --output_dir=$OUT_DIR --phase_portrait

python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot2="lossrd/dyn_beta_y" --plot1="loss_task/ell" --legend2="KL (beta_y)" --legend1="ell" --plot_len $phase_portrait_plot_len --skip_n_steps $skip_n --output_dir=$OUT_DIR --phase_portrait




# Plot R and the corresponding set point curves (both in the same figure)
python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot1="lossrd/dyn_gamma_d" --plot2="lossrs/setpoint_gamma_d" --legend1="\$R_{\gamma_d}\$" --legend2="setpoint" --output_dir=$OUT_DIR

python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot1="lossrd/dyn_mu_recon" --plot2="lossrs/setpoint_mu_recon" --legend1="reconstruction loss" --legend2="setpoint" --output_dir=$OUT_DIR

python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot1="lossrd/dyn_beta_d" --plot2="lossrs/setpoint_beta_d" --legend1="-\$R_{\beta_d}\$" --legend2="setpoint" --output_dir=$OUT_DIR --neg

# python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot1="lossrd/dyn_beta_x" --plot2="lossrs/setpoint_beta_x" --legend1="KL (beta_x)" --legend2="setpoint" --output_dir=$OUT_DIR

python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot1="lossrd/dyn_beta_y" --plot2="lossrs/setpoint_beta_y" --legend1="-\$R_{\beta_y}\$" --legend2="setpoint" --output_dir=$OUT_DIR --neg


 # One curve per figure
 values=('controller_gain/beta_d' 'controller_gain/beta_y' 'controller_gain/beta_x' 'controller_gain/gamma_d' 'controller_gain/mu_recon' 'dyn_mu/beta_d' 'delta/beta_d' 'dyn_mu/beta_y' 'delta/beta_y' 'dyn_mu/beta_x' 'delta/beta_x' 'dyn_mu/gamma_d' 'delta/gamma_d' 'dyn_mu/mu_recon' 'delta/mu_recon' 'loss_task/penalized' 'loss_task/ell' 'acc/te' 'acc/val' 'acc/sel' 'acc/setpoint')
 # Loop over the array
 for val in "${values[@]}"
 do
   python domainlab/utils/generate_fbopt_phase_portrait.py --runs_dir $runs_dir --plot1="$val" --legend1="$val" --output_dir=$OUT_DIR
 done
