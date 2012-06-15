#!/bin/sh

octave --eval "

% Produce a bitmap with the desired pattern

X = zeros(128); X(30, 30) = 1;


imwrite(X, 'test.bmp');

% Call DTCWT verification on bitmap; it will return test.bmp.0.0,
% test.bmp.0.1 etc for increasing subbands (the other number is for wavelet 
% level).
system('./verifyBasic test.bmp');

[Yl, Yh] = dtwavexfm2b(X, 3, 'near_sym_b_bp', 'qshift_b_bp');

idx = [3 5 1 6 2 4];
for l = 1:3
    for sb = 1:6

        % Read the data
        vbY = dlmread(sprintf('test.bmp.%d.%d', l-1, sb-1), ',');

        % Compare to reference implementation
        difference = abs(Yh{l}(:,:,idx(sb)) - vbY);

        % Check it's all close enough to right
        if any(difference(:) > 1e-3)
            disp('DTCWT failed to meet tolerances!!!');

            % Flag an error
            quit(1);
        end

    end
end

display('DTCWT worked!')
quit(0)
"

# Return the same code octave did
exit $?

