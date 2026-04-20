clear;
close all;
clc;

codeDir = fileparts(mfilename('fullpath'));
if isempty(codeDir)
    codeDir = pwd;
end
graphDir = fullfile(fileparts(codeDir), 'graph');
if ~exist(graphDir, 'dir')
    mkdir(graphDir);
end

T = 2;
w0 = 2 * pi / T;
numSamples = 6000;
kMax = 25;
k = (-kMax:kMax).';

tPeriod = linspace(-1, 1, numSamples + 1);
tPeriod(end) = [];
xPeriod = 1 - abs(tPeriod);

aTheory = zeros(size(k));
for idx = 1:numel(k)
    aTheory(idx) = problem3_coefficient(k(idx));
end

aEmpirical = zeros(size(k));
for idx = 1:numel(k)
    aEmpirical(idx) = mean(xPeriod .* exp(-1i * k(idx) * w0 * tPeriod));
end

tMulti = linspace(-5, 5, 4000);
tau = mod(tMulti + 1, T) - 1;
xMulti = 1 - abs(tau);

signalFig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1, 1, 5.46, 1.75]);
plot(tMulti, xMulti, 'LineWidth', 1.2);
grid on;
xlabel('t');
ylabel('x(t)');
title('Problem 3: Periodic Triangular Signal with T = 2');
exportgraphics(signalFig, fullfile(graphDir, 'problem3_signal.png'), 'Resolution', 300);
close(signalFig);

coeffFig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1, 1, 5.46, 1.75]);
stem(k, abs(aTheory), 'filled', 'LineWidth', 1.2, 'DisplayName', 'Theoretical'); hold on;
stem(k + 0.15, abs(aEmpirical), 'LineWidth', 1.0, 'DisplayName', 'Empirical');
grid on;
xlabel('k');
ylabel('|a_k|');
title('Problem 3: Fourier Series Coefficients');
legend('Location', 'northeast');
exportgraphics(coeffFig, fullfile(graphDir, 'problem3_coefficients.png'), 'Resolution', 300);
close(coeffFig);

NValues = [1, 9, 15];
reconFig = figure('Visible', 'off', 'Color', 'w');
tiledlayout(numel(NValues), 1, 'Padding', 'compact', 'TileSpacing', 'compact');
for idx = 1:numel(NValues)
    N = NValues(idx);
    xRecon = reconstruct_from_coefficients(k, aEmpirical, w0, tMulti, N);
    nexttile;
    plot(tMulti, xMulti, 'k', 'LineWidth', 1.6, 'DisplayName', 'Original'); hold on;
    plot(tMulti, real(xRecon), 'r--', 'LineWidth', 1.3, 'DisplayName', sprintf('Reconstruction (N=%d)', N));
    grid on;
    ylabel('x(t)');
    title(sprintf('Problem 3 Reconstruction with N = %d', N));
    if idx == 1
        legend('Location', 'best');
    end
end
xlabel('t');
exportgraphics(reconFig, fullfile(graphDir, 'problem3_reconstruction.png'), 'Resolution', 300);
close(reconFig);

function coeff = problem3_coefficient(k)
if k == 0
    coeff = 1 / 2;
elseif mod(abs(k), 2) == 0
    coeff = 0;
else
    coeff = 2 / (pi^2 * k^2);
end
end

function xRecon = reconstruct_from_coefficients(k, coeffs, w0, t, N)
mask = abs(k) <= N;
xRecon = zeros(size(t));
for idx = find(mask).'
    xRecon = xRecon + coeffs(idx) .* exp(1i * k(idx) * w0 * t);
end
end
