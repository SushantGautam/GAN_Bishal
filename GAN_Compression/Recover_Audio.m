function [f_amp_pgla] = Recover_Audio(spectrum,a,M,tfr)
%RECOVER_AUDIO Summary of this function goes here
%   Detailed explanation goes here
tfdata_amp = spectrum;
win = {'gauss', tfr};
flag = 'freqinv';
Ltrue =  size(tfdata_amp,2)*a;
win = gabwin(win,a,M,Ltrue);
dual = gabdual(win,a,M,Ltrue);
gamma = pghi_findgamma(win,a,M,Ltrue);
mask = zeros(M/2+1, size(tfdata_amp, 2));
c_amp_pghi = pghi(tfdata_amp,gamma,a,M,mask,flag);
f_amp_pgla = idgtreal(c_amp_pghi,dual,a,M,flag);
end

