%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is used to generate CIR
% 
% Author: Mingyu Yang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

if ~exist('traindata', 'dir')
    mkdir('traindata')
end

load('Pathset_train.mat')

DISPLAY      = 0;            % 1 for display mode, 0 for generation mode
ofdm_bw      = 40e6;         % Target ofdm bandwidth
upsample     = 2;            % Up-sampling rate for super-resolution

% *_A for CIR enhancement and *_B for ToA estimation stage
saved_dist_A = saved_dist(1:100000);
saved_dist_B = saved_dist(100001:end);
saved_mag_A = saved_mag(1:100000);
saved_mag_B = saved_mag(100001:end);
saved_paths_A = saved_paths(1:100000);
saved_paths_B = saved_paths(100001:end);

%%%%%%%%%%%%%% Generate low SNR training sets %%%%%%%%%%%%%%%%%%
SNR_list       = [-2.5:0.1:7.5];        % Set of possible SNR (dB)
save_path  = ['traindata/Train_x', num2str(upsample), '_low_', num2str(ofdm_bw*1e-6), 'MHz_A.mat'];
generate_dataset(saved_dist_A,saved_mag_A,saved_paths_A,SNR_list,ofdm_bw,upsample,save_path,DISPLAY)
save_path  = ['traindata/Train_x', num2str(upsample), '_low_', num2str(ofdm_bw*1e-6), 'MHz_B.mat'];
generate_dataset(saved_dist_B,saved_mag_B,saved_paths_B,SNR_list,ofdm_bw,upsample,save_path,DISPLAY)

%%%%%%%%%%%%%% Generate high SNR training sets %%%%%%%%%%%%%%%%%%
SNR_list       = [7.5:0.1:32.5];        % Set of possible SNR (dB)
save_path  = ['traindata/Train_x', num2str(upsample), '_high_', num2str(ofdm_bw*1e-6), 'MHz_A.mat'];
generate_dataset(saved_dist_A,saved_mag_A,saved_paths_A,SNR_list,ofdm_bw,upsample,save_path,DISPLAY)
save_path  = ['traindata/Train_x', num2str(upsample), '_high_', num2str(ofdm_bw*1e-6), 'MHz_B.mat'];
generate_dataset(saved_dist_B,saved_mag_B,saved_paths_B,SNR_list,ofdm_bw,upsample,save_path,DISPLAY)

%%
%%%%%%%%%%%%% Generate test sets %%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('testdata', 'dir')
    mkdir('testdata')
end
    
load('Pathset_test.mat')
DISPLAY      = 0;            % 1 for display mode, 0 for generation mode
ofdm_bw      = 40e6;         % Target ofdm bandwidth
upsample     = 2;
SNR_set      = [0:5:30];

for i = 1:length(SNR_set)
    SNR_list  = [SNR_set(i), SNR_set(i)];        % Set of possible SNR (dB)
    save_path  = ['testdata/Test_x', num2str(upsample), '_', num2str(SNR_set(i)), 'dB_',num2str(ofdm_bw*1e-6),'MHz.mat'];
    generate_dataset(saved_dist,saved_mag,saved_paths,SNR_list,ofdm_bw,upsample,save_path,DISPLAY)
end

load('Pathset_test_802.mat')
DISPLAY      = 0;            % 1 for display mode, 0 for generation mode
ofdm_bw      = 40e6;         % Target ofdm bandwidth
upsample     = 2;
SNR_set      = [0:5:30];

for i = 1:length(SNR_set)
    SNR_list  = [SNR_set(i), SNR_set(i)];        % Set of possible SNR (dB)
    save_path  = ['testdata/Test_x', num2str(upsample), '_', num2str(SNR_set(i)), 'dB_802_',num2str(ofdm_bw*1e-6),'MHz.mat'];
    generate_dataset(saved_dist,saved_mag,saved_paths,SNR_list,ofdm_bw,upsample,save_path,DISPLAY)
end


