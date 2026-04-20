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

T = 3;
w0 = 2 * pi / T;
numSamples = 6000;
kMax = 20;
k = (-kMax:kMax).';

tPeriod = linspace(0, T, numSamples + 1);
tPeriod(end) = [];
xPeriod = exp(-tPeriod);

aTheory = (1 - exp(-T)) ./ (T * (1 + 1i * k * w0));
aEmpirical = zeros(size(k));
for idx = 1:numel(k)
    aEmpirical(idx) = mean(xPeriod .* exp(-1i * k(idx) * w0 * tPeriod));
end

tMulti = linspace(-2 * T, 2 * T, 4000);
xMulti = exp(-mod(tMulti, T));

signalFig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1, 1, 5.46, 1.75]);
plot(tMulti, xMulti, 'LineWidth', 1.2);
grid on;
xlabel('t');
ylabel('x(t)');
title('Problem 1: Periodic Signal x(t) = e^{-t}, 0 \leq t < 3');
exportgraphics(signalFig, fullfile(graphDir, 'problem1_signal.png'), 'Resolution', 300);
close(signalFig);

coeffFig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1, 1, 5.46, 1.75]);
stem(k, abs(aTheory), 'filled', 'LineWidth', 1.2, 'DisplayName', 'Theoretical'); hold on;
stem(k + 0.15, abs(aEmpirical), 'LineWidth', 1.0, 'DisplayName', 'Empirical');
grid on;
xlabel('k');
ylabel('|a_k|');
title('Problem 1: Fourier Series Coefficients');
legend('Location', 'northeast');
exportgraphics(coeffFig, fullfile(graphDir, 'problem1_coefficients.png'), 'Resolution', 300);
close(coeffFig);

NValues = [3, 10, 20];
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
    title(sprintf('Problem 1 Reconstruction with N = %d', N));
    if idx == 1
        legend('Location', 'best');
    end
end
xlabel('t');
exportgraphics(reconFig, fullfile(graphDir, 'problem1_reconstruction.png'), 'Resolution', 300);
close(reconFig);

function xRecon = reconstruct_from_coefficients(k, coeffs, w0, t, N)
mask = abs(k) <= N;
xRecon = zeros(size(t));
for idx = find(mask).'
    xRecon = xRecon + coeffs(idx) .* exp(1i * k(idx) * w0 * t);
end
end
