filenames = {'TAPE29.txt', 'TAPEFILL.txt', 'TAPEGAP.txt'};

legendnames = {'Tape over', 'Tape over foam insert', 'Tape over foam insert with channels'};
data = cell(size(filenames));

for ii = 1:numel(filenames)
    data{ii} = csvread(filenames{ii});
end


offset_t = [0, 0, -75, 0, 0];
offset_p = [0, 0, -2, 0, 0];


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
    yy = data{ii} + offset_p(ii);
    xx = (1:numel(yy)) + offset_t(ii);
    plot(xx, yy)
end
xlim([0, 1400])
ylim([5, 45])

legend(legendnames, 'location', 'SE', 'fontsize', 10)
xlabel('Time (seconds)')
ylabel('O2 Concentration (uncalibrated)')
saveas(gca, 'foam_insert.jpg')

xlim([400, 700])
ylim([5, 45])
saveas(gca, 'foam_insert_zoom.jpg')
