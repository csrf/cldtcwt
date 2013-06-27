% Generates the constants needed by kernel.cl

w = [-3 -1;
     -sqrt(5) -sqrt(5);
     -1 -3;
     1 -3;
     sqrt(5) -sqrt(5);
     3 -1]' * pi / 2.15;

% Rotate to sampling positions and normalise the length
R = [0 1; -1 0];
d = R * w;
d = bsxfun(@rdivide, d, sqrt(sum(d.^2, 1)));

% Calculate positions of each of the sampling points
a0Pos = floor(d);
a1Pos = bsxfun(@plus, a0Pos, [0 1]');
a2Pos = bsxfun(@plus, a0Pos, [1 0]');
a3Pos = bsxfun(@plus, a0Pos, [1 1]');

% Apply linear interpolation and derotation (for bandpass)
a0Coeff = prod(1 - abs(d - a0Pos), 1) .* exp(-j * sum(w .* a0Pos, 1));
a1Coeff = prod(1 - abs(d - a1Pos), 1) .* exp(-j * sum(w .* a1Pos, 1));
a2Coeff = prod(1 - abs(d - a2Pos), 1) .* exp(-j * sum(w .* a2Pos, 1));
a3Coeff = prod(1 - abs(d - a3Pos), 1) .* exp(-j * sum(w .* a3Pos, 1));

coeff = [a0Coeff; a1Coeff; a2Coeff; a3Coeff].';

% Convert to C
disp('{');
for s = 1:size(coeff, 1)
    disp('    {');
    for n = 1:size(coeff, 2)

        if n == size(coeff,2)
            disp(sprintf('        (Complex) (%ff,%ff)',...
                         real(coeff(s,n)), imag(coeff(s,n))));
        else
            disp(sprintf('        (Complex) (%ff,%ff),',...
                         real(coeff(s,n)), imag(coeff(s,n))));
        end

    end

    if s == size(coeff,1)
        disp('    }');
    else
        disp('    },');
    end
end
disp('}');

