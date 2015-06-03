function [P] = get_spectrogram_orig(Y, padding, fs)

% if ~exist('fs', 'var')
% 	warning('sample rate was not specified: using the rate for TIMIT instead.. If the audio file is not from TIMIT corpus, you should set this value correctly!!');
%     fs = get_constant('TimitSampleRate');
% end

wintime = get_constant('TimitWindowTime');
hoptime = get_constant('TimitHopTime');
nfft = ceil(fs*wintime);
WINDOW = hamming(nfft);
noverlap = nfft-ceil(fs*hoptime);

freq_low = get_constant('TimitFreqLow');
freq_high = get_constant('TimitFreqHigh');

if ~exist('padding', 'var')
    padding = get_constant('SpectrogramPadding');
end

[P,~] = spectrogram(Y, WINDOW, noverlap, nfft, fs);
P = flipud(abs(P));

P = log(1e-5+P) - log(1e-5);

P = P - mean(mean(P));

P = P / sqrt(mean(mean(P.^2)));

P = [zeros(size(P,1),padding) smooth(P',20,4)' zeros(size(P,1),padding)];
%P = [zeros(size(P,1),padding) smooth(P',20,'sgolay')' zeros(size(P,1),padding)];

