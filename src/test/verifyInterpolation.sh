#!/bin/sh

octave --eval "

% Produce a bitmap with the desired pattern

X = zeros(128); X(128, 128) = 255;


imwrite(X, 'test.bmp');

% Call DTCWT verification on bitmap; it will return test.bmp.0.0,
% test.bmp.0.1 etc for increasing subbands (the other number is for wavelet 
% level).
system('./testDescriptor test.bmp');

[Yl, Yh] = dtwavexfm2b(X, 3, 'near_sym_b_bp', 'qshift_b_bp');

% Design the pattern
z = exp(i * (0:11)' / 12 * 2 * pi);
pattern = [0 0; real(z) imag(z)];


locs = bsxfun(@plus, [28.5 28.75], pattern)

% Correct the phases and perform band-pass interpolation
Yh = correctPhase(Yh);

r5 = sqrt(5); 
w = [-3 -1; -r5 -r5; -1 -3; 1 -3; r5 -r5; 3 -1]*pi/2.15; 
% Nominally pi/2, but reduced a bit due to asymmetry of subband freq responses.

% Generate reference outputs
ref = zeros(6,size(pattern,1));
for n = 1:6
    for m = 1:size(pattern,1);
        ref(n,m) = transpose(cpxinterp2b(Yh{2}(:,:,n), locs(m,:), w(n,:)));
    end
end

out = dlmread('interpolations.dat', ',');

    disp('Should have been:')
    ref
    disp('Was:')
    reshape(out, 6, numel(out) / 6)

if any(abs(out(1:numel(ref)) - ref(:)) > 1e-5)
    disp('Interpolations did not give the same results as Octave!')
    quit(1)
else
    display('Interpolation worked!')
end


quit(0)
"

# Return the same code octave did
exit $?

