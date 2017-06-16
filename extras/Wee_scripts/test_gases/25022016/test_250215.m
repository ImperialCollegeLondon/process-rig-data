filenames = {'TAPE25.txt', 'TAPEBOT.txt', 'PLATESI.txt', 'PLATET.txt', 'PLATET2.txt'};
legendnames = {'Tape over', 'Tape on the bottom', 'Plates Silicon Grease Side (no wall)', ...
    'Tape plate sides 1', 'Tape plate sides 2'};
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


xlim([0, 1200])