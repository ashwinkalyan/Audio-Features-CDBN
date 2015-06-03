function [poshidprobs] = get_poshidprobs(speechpath,W, hbias_vec, pars, spacing,Ewhiten)

% vbias_vec not required!!?

[y,fs]=wavread(speechpath);
P2 = get_spectrogram_orig(y, 0, fs);
ws = size(W,1);

imdata = Ewhiten*P2;
imdata = trim_audio_for_spacing_fixconv(imdata, ws, spacing);
imdatatr = imdata';
imdatatr = reshape(imdatatr, [size(imdatatr,1), 1, size(imdatatr,2)]);

poshidexp = tirbm_inference_fixconv_1d(imdatatr, W, hbias_vec, pars);
[~,poshidprobs] = tirbm_sample_multrand_1d(poshidexp, spacing);