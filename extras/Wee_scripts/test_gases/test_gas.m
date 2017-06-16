aa = csvread('bot.txt');
xx = 850:1125;
yy = aa(xx);

figure
plot(yy)
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)

%%
bb = csvread('plat.txt');
figure,
plot(bb)
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)


cc = csvread('plat_bot.txt');
figure,
plot(cc)
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)

part1 = 340:710;
part2 = 850:1370;
part3 = 1430:2490;

%%
figure
%bottom plate NO holes
plot(part1, cc(part1))

figure
%bottom sensor near the inlet
plot(part2, cc(part2))

figure
%bottom plate holes
plot(part3, cc(part3))

%%
yy_inlet = csvread('161015_Inlet.txt');
yy_inlet = yy_inlet(1350:end);

yy_plate = csvread('191015_PlateCam2.txt');
yy_plate = yy_plate(1:28800);


tt_inlet = (0:numel(yy_inlet)-1);%/60;
%%
figure, hold on
vv = 1:1885;
plot(tt_inlet(vv), yy_inlet(vv))
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)
%%
tot = numel(yy_inlet);
figure, hold on
vv = 1865:tot;
tt = (0:numel(vv)-1)/60; 
plot(tt, yy_inlet(vv))

ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [minutes]', 'fontsize', 12)
ylim([0 70])
xlim([0, 105])
%%
tot = numel(yy_inlet);
figure, hold on
vv = 1865+(1:(60*42));%numel(tyyy_inlety_inlett);
plot( yy_inlet(vv))
vv = 4305:tot;
plot( yy_inlet(vv))

ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [minutes]', 'fontsize', 12)
ylim([0 70])
%%
figure
vv = 1:6515;
tt_plate = (0:numel(yy_plate)-1);%/60;

plot(tt_plate(vv), yy_plate(vv))
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [minutes]', 'fontsize', 12)
%%
tot = numel(tt_plate);

figure, hold on
vv = 6190:tot;
tt = (0:numel(vv)-1)/60; 
plot(tt, yy_plate(vv))

ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [minutes]', 'fontsize', 12)
ylim([0 70])
xlim([0 380])

%%
tot = numel(tt_plate);

figure, hold on

vv = 6190+(1:(4*60*42));%numel(tyyy_inlety_inlett);
plot(yy_plate(vv))
vv = 15930:tot;%numel(tyyy_inlety_inlett);
plot(yy_plate(vv))

ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [minutes]', 'fontsize', 12)
ylim([0 70])

%%
figure, hold on
yi = yy_inlet(590:800);
xi = (1:numel(yi)) - 95;
plot(xi, yi)

yp = yy_plate(2550:2900);
xp = (1:numel(yp)) - 135;
plot(xp, yp)

legend({'Inlet', 'Plate'}, 'location', 'SE', 'fontsize', 16)
xlim([-100, 200])
ylabel('Sensor O2[%]', 'fontsize', 14)
xlabel('Time [seconds]', 'fontsize', 14)
%%
figure, hold on
yi = yy_inlet(720:930);
xi = (1:numel(yi)) - 85;
plot(xi, yi)

yp = yy_plate(2900:3500);
xp = (1:numel(yp)) - 212;
plot(xp, yp)

legend({'Inlet', 'Plate'}, 'location', 'SE', 'fontsize', 16)
xlim([-100, 200])
ylabel('Sensor O2[%]', 'fontsize', 14)
xlabel('Time [seconds]', 'fontsize', 14)

%%
hh = csvread('H5_TEST.txt');
tt = (0:numel(hh)-1)/60-43;
figure, hold on
plot(tt, hh)
ylabel('Sensor O2[%]', 'fontsize', 14)
xlabel('Time [minutes]', 'fontsize', 14)


hh = csvread('A021115.txt');
%maybe here there was high humidity in the chamber, and affected the O2
%reading.
tt = (0:numel(hh)-1)/60*5;
plot(tt, hh)
ylabel('Sensor O2[%]', 'fontsize', 14)
xlabel('Time [minutes]', 'fontsize', 14)

hh = csvread('A031115.txt');
%the sensor seemed to have got loose, and the recording was interupted due
%because the meter runout of battery
tt = (0:numel(hh)-1)/60*5-10;
plot(tt, hh)
ylabel('Sensor O2[%]', 'fontsize', 14)
xlabel('Time [minutes]', 'fontsize', 14)


grid on

%%
yy = csvread('A231115.txt');

figure
plot(yy)
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)
%%
yy = csvread('A241115R_1.txt');

figure
plot(yy)
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)
%%
yy = csvread('A241115R_2.txt');

figure
plot(yy)
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)

%%
yy = csvread('PLAT_BOT.txt');



figure, hold on
plot(yy)
ylabel('Sensor O2[%]', 'fontsize', 12)
xlabel('Time [seconds]', 'fontsize', 12)
ylim([0, 50])
plot(isnan(yy)*50)
%xlim([7000, 15000])

%%
several_xlim = {[0, 900]};

for ii = 1:numel(several_xlim)
    figure
    plot(yy)
    ylabel('Sensor O2[%]', 'fontsize', 12)
    xlabel('Time [seconds]', 'fontsize', 12)
    ylim([0, 50])
    xlim(several_xlim{ii})
end
