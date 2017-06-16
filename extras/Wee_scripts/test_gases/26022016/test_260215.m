filenames = {'TAPE26.txt', 'TAPEBOTF.txt', 'TAPE262.txt', 'TBOT2.txt'};
legendnames = {'Tape over 1', 'Tape on the bottom and sides 1', ...
    'Tape over 2', 'Tape on the bottom and sides 2'};
data = cell(size(filenames));

for ii = 1:numel(filenames)
    data{ii} = csvread(filenames{ii});
end


offset = [0, 0, 0, 0, 0];
figure, hold on
for ii = 1:numel(filenames)
    yy = data{ii};
    xx = (1:numel(yy)) + offset(ii);
    plot(xx, yy)
end
legend(legendnames, 'location', 'NW')
xlim([0, 1200])

%%
filenames = {'TAPE263.txt', 'TAPE264.txt'};
legendnames = {'Tape over bad seal', 'Tape over good seal'};
data = cell(size(filenames));

for ii = 1:numel(filenames)
    data{ii} = csvread(filenames{ii});
end


offset = [0, 0, 0, 0, 0];

h = figure;
hold on
box on
set(h,'paperunits','centimeters')
set(gca,'fontsize', 16, 'fontname','arial')
set(gca,'position', [0.2 0.15 0.75 0.75])
set(gca,'ticklength',[0.02 0.015])
set(h,'papersize', [14.5 11])
set(h,'paperposition', [0 0 14.5 11])
numFont = 20;

for ii = 1:numel(filenames)
    yy = data{ii};
    xx = (1:numel(yy)) + offset(ii);
    plot(xx, yy)
end
legend(legendnames, 'location', 'SE', 'fontsize', 10)
xlabel('Time (seconds)')
ylabel('O2 Concentration (uncalibrated)')
saveas(gca, 'bad_seal.jpg')
%%
filenames = {'RAMPTAPE.txt'};
legendnames = {'Ramp'};
data = cell(size(filenames));

for ii = 1:numel(filenames)
    data{ii} = csvread(filenames{ii});
end


offset = [0, 0, 0, 0, 0];
figure, hold on
for ii = 1:numel(filenames)
    yy = data{ii};
    xx = (1:numel(yy)) + offset(ii);
    plot(xx, yy)
end
legend(legendnames, 'location', 'NW')