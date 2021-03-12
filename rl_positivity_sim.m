%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation of RL deconvolution under noise-free condition            %
% Adapted from James Manton                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Simulation parameters
mid_bg = 0.05;
right_bg = 0.25;
sigma = 5;
iteration = 5000;
n = 512;
spacing_px = 4;

SAVE_DISPLAY = 1;
USE_GPU = 0;

%% Create line pairs and add background levels
x = zeros(n);
x((n/4):(3*n/4), (n/2) - spacing_px) = 1;
x((n/4):(3*n/4), (n/2) + spacing_px) = 1;
x = x + circshift(x, [0, round(n/3)]) + circshift(x, [0, -round(n/3)]);

x(7*n/8 - spacing_px, (n/4):(3*n/4)) = 1;
x(7*n/8 + spacing_px, (n/4):(3*n/4)) = 1;

x(n/2 - spacing_px, (n/16):(15*n/16)) = 1;
x(n/2 + spacing_px, (n/16):(15*n/16)) = 1;

x(:, round(n/3):round(2*n/3)) = x(:, round(n/3):round(2*n/3)) + mid_bg;
x(:, round(2*n/3):end) = x(:, round(2*n/3):end) + right_bg;

%% Create OTF
A=generate_psf(sigma);
otfA=psf2otf(A,size(x));

%% Simulate captured data
y = real(ifft2(fft2(x) .* otfA));
y = y ./ max(y(:));

%% Deconvolve data
RL=deconvlucy(y,A,iteration);
RL = RL ./ max(RL(:));

%% Calculate spectra
spectrum_field = log(1 + abs(fftshift(fft2(x))));
spectrum_field = spectrum_field ./ max(spectrum_field(:));
spectrum_rl = log(1 + abs(fftshift(fft2(RL))));
spectrum_rl = spectrum_rl ./ max(spectrum_rl(:));

%% Display (and save) results
figure(1)
display_array = [x ./ max(x(:)), spectrum_field, fftshift(otfA); ...
    y ./ max(y(:)), RL, spectrum_rl];
imshow(display_array, [])

if SAVE_DISPLAY
    imwrite(display_array, 'rl_positivity_sim.png')
end

function y=generate_psf(gama)
gama=gama/2.335;
sigma=max(gama,gama);
kernelRadius = ceil(sigma* sqrt(-2* log(0.0002)))+1;
ii=-kernelRadius:kernelRadius;
rsf_x=1/2*(erf((ii+0.5)./(sqrt(2).*gama))-erf((ii-0.5)./(sqrt(2).*gama)));
kernel= rsf_x'* rsf_x;
y=kernel./sum(kernel(:));
end