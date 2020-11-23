
function [freq_resp] = sum_exponentials(freq_grid,paths,coef)

N = length(paths);
freq_resp = zeros(size(freq_grid));
for i = 1:N
    freq_resp = freq_resp + coef(i)*exp(-1j*2*pi*freq_grid*paths(i));
end

end

