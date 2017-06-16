filenames = {'SI1.txt', 'SI2.txt', 'TAPE1.txt', 'FULL1.txt'};
legendnames = {'Silicon Grease Plates', 'Silicon Grase Border', 'Tape over plates', 'Plates with wall'};
data = cell(size(filenames));

for ii = 1:numel(filenames)
    data{ii} = csvread(filenames{ii});
end


offset = [-20, 0, 20, -20];
figure, hold on
for ii = 1:numel(filenames)
    yy = data{ii};
    xx = (1:numel(yy)) + offset(ii);
    plot(xx, yy)
end
legend(legendnames, 'location', 'NW')
xlim([0, 1200])