filenames = {'PLATESI.txt', 'TAPE4.txt', 'TAPE4_2.txt', 'TAPEFILL.txt', 'TAPECH.txt'};
legendnames = {'Plates Silicon Grease Side', 'Tape over plates 1', 'Tape over plates 2', ...
    'Tape over plates Filled', 'Tape over plates Channel'};
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