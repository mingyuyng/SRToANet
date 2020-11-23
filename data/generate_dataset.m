function generate_dataset(saved_dist,saved_mag,saved_paths,SNR_list, ofdm_bw,upsample,path,debug)


DISPLAY      = debug;                        % 1 for display mode, 0 for generation mode

tones_gap    = 312.5e3;                      % 802.11 g/n/ac standard
N_tones      = round(ofdm_bw / tones_gap);   % number of subcarriers        

signal_pwr_dB  = 10;
signal_pwr     = 10^((signal_pwr_dB)/10);

C = 3e8;   % speed of light

%%%%%%%%%%%%%%% Generate dataset %%%%%%%%%%%%%%%%%%%%%%%

N = length(saved_dist);
cir_l = zeros(N, 2, N_tones*upsample);
cir_h = zeros(N, 2, N_tones*upsample);
cfr_h = zeros(N, 2, N_tones*upsample);

fprintf(['Generating ', path, '...,  Upsample: %d, Bandwidth: %d\n'], upsample, ofdm_bw/1e6);

for kk = 1:length(saved_dist)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Channel Delay Spread Setting
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SNR             = randsample(SNR_list,1);
    noise_pwr_dB    = signal_pwr_dB-SNR;
    noise_pwr       = 10^((noise_pwr_dB)/10);
                                                        
    % Sampling points for small and large bands
    dd = saved_dist(kk);
        
    % Sampling grid for the channel frequency response
    grid_l = ofdm_bw * (-1/2 : 1/N_tones : 1/2 - 1/N_tones);
    grid_h = ofdm_bw*upsample * (-1/2 : 1/N_tones/upsample : 1/2 - 1/N_tones/upsample);
    
    % load the path information
    paths = saved_paths{kk};
    coef  = saved_mag{kk};
    
    % Sample the ground truth channel frequency response
    freq_resp_l = fftshift(sum_exponentials(grid_l, paths*1e-9, coef));
    freq_resp_h = fftshift(sum_exponentials(grid_h, paths*1e-9, coef));
    
    % Normalization to unit power
    power = mean(abs(freq_resp_l).^2);
    freq_resp_l = sqrt(signal_pwr)*freq_resp_l/sqrt(power);
    freq_resp_h = sqrt(signal_pwr)*freq_resp_h/sqrt(power);
    
    % add noise
    noise = sqrt(noise_pwr/2) * (randn(1,N_tones) + 1j*randn(1,N_tones));
    freq_resp_obs  = freq_resp_l + noise;
    
    % Zero padding the observed channel frequency response
    freq_resp_obs_pad = [freq_resp_obs(1:N_tones/2), zeros(1,(upsample-1)*N_tones), freq_resp_obs(end-N_tones/2+1:end)];
    
   
    if DISPLAY == 1
           
        time_h = [0:N_tones*upsample-1]/ofdm_bw/2;
        dist_h = time_h*C;

        plot(dist_h,abs(ifft(freq_resp_obs_pad)));
        xlabel("Distance")
        title(['SNR (dB):', num2str(SNR), '  Badnwidth (MHz):', num2str(ofdm_bw/1e6)])
        hold on;
        plot(dist_h,abs(ifft(freq_resp_h(:))));
        hold on;
        plot(dd*ones(1,100),linspace(0,1.25*max(abs(ifft(freq_resp_obs_pad))),100));
        legend('low resolution', 'high resolution', 'ground truth')

        figure;

        plot(grid_h,abs(fftshift(freq_resp_obs_pad)));
        xlabel("Distance")
        title(['SNR (dB):', num2str(SNR), '  Badnwidth (MHz):', num2str(ofdm_bw/1e6)])
        hold on;
        plot(grid_h,abs(fftshift(freq_resp_h(:))));
        hold on;
        legend('low resolution', 'high resolution')
        keyboard;
    end
    
    
    cir_pad = ifft(freq_resp_obs_pad(:));
    cirh = ifft(freq_resp_h(:));
    cfrh = freq_resp_h(:);
 
    cir_l(kk,1,:) = real(cir_pad);
    cir_l(kk,2,:) = imag(cir_pad);
    cfr_h(kk,1,:) = real(cfrh);
    cfr_h(kk,2,:) = imag(cfrh);
    cir_h(kk,1,:) = real(cirh);
    cir_h(kk,2,:) = imag(cirh);
    
    
end

dist = saved_dist;
save(path, 'cir_l', 'cir_h','cfr_h', 'dist');  
    
end

