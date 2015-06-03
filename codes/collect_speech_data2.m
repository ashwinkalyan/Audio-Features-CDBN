function Pall = collect_speech_data2(data_root)

% initialize
Pall = {};

% get all the speech files from data_root
flist = dir(fullfile(strcat(data_root,'*.wav')));
%flist= get_speech_filenames2();

parfor i=1:length(flist)
    [y,fs]=wavread(strcat(data_root,flist(i).name));
   %[~,P2,~] = load_spectrogram(flist(i).name);
    P2 = get_spectrogram_orig(y, 0, fs);
    fprintf('done with \t: %d \n',i);
    Pall{i} = P2;
end
    
% parfor i= 1:length(flist)
%     if mod(i,10)==0, fprintf('.'); end
%     if mod(i,1000)==0, fprintf('\n'); end
%     [P, P2, y] = load_spectrogram(flist{i});
%     Pall{i} = P2;
% end










