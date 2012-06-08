
% Set up the input in the same way as in verifyBasic
input = zeros(128,128);
input(64,64) = 1;

[Yl, Yh, Yscale, lohi] = dtwavexfm2b(input, 6, 'near_sym_b_bp', 'qshift_b_bp');



