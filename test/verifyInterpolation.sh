#!/bin/sh

octave --eval "

% Produce a bitmap with the desired pattern

X = zeros(128); X(128, 128) = 1;


imwrite(X, 'test.bmp');

% Call DTCWT verification on bitmap; it will return test.bmp.0.0,
% test.bmp.0.1 etc for increasing subbands (the other number is for wavelet 
% level).
system('./testDescriptor test.bmp');

[Yl, Yh] = dtwavexfm2b(X, 3, 'near_sym_b_bp', 'qshift_b_bp');

locs = [32 31];

% Correct the phases and perform band-pass interpolation
Yh = correctPhase(Yh);

r5 = sqrt(5); 
w = [-3 -1; -r5 -r5; -1 -3; 1 -3; r5 -r5; 3 -1]*pi/2.15; 
% Nominally pi/2, but reduced a bit due to asymmetry of subband freq responses.

out = zeros(6,1);
for n = 1:6
    out(n) = cpxinterp2b(Yh{2}(:,:,n), locs, w(n,:));
end

out

% Read the data
%vbY = dlmread(sprintf('test.bmp.%d.%d', l-1, sb-1), ',');

display('DTCWT worked!')
quit(0)
"

# Return the same code octave did
exit $?

