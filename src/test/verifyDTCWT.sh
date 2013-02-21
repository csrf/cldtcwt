#!/bin/sh

octave --silent --eval "

% Produce a bitmap with the desired pattern

X = zeros(256); 


for n = 10:100;
    X(n-1,n) = 1;
    X(n,n) = 1;
    X(n+1,n) = 1;
end



imwrite(X, 'testDTCWT.bmp');

% Call DTCWT verification on bitmap; it will return test.bmp.0.0,
% test.bmp.0.1 etc for increasing subbands (the other number is for wavelet 
% level).
system('./verifyBasic testDTCWT.bmp');

[Yl, Yh] = dtwavexfm2b(X, 3, 'near_sym_b_bp', 'qshift_b_bp');

for l = 1:3
    for sb = 1:6

        % Read the data
        vbY = dlmread(sprintf('testDTCWT.bmp.%d.%d', l-1, sb-1), ',');

        % Compare to reference implementation
        difference = abs(Yh{l}(:,:,sb) - vbY);

        disp(sprintf('%d %d %f', l, sb, max(difference(:))));

        % Check it's all close enough to right
        if any(difference(:) > 5e-3)
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

